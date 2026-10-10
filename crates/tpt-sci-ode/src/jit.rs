//! Cranelift-generated call stubs for right-hand-side closures.
//!
//! [`compile_rhs`] wraps a Rust closure in a native function with the C ABI
//! `fn(ctx, t, y, dydt)`. That function is a thin stub: it forwards its
//! arguments to a Rust trampoline, which runs the closure. The closure body is
//! ordinary Rust and is not compiled to machine code. The stub gives an
//! ABI-compatible function pointer, not a speed-up over calling the closure
//! directly. Compiling expressions (for example from a DiffSL description) to
//! native code is future work.
//!
//! # Ownership
//!
//! A [`JitRhs`] owns both the closure and the Cranelift module that holds the
//! stub's machine code, so the function pointer stays valid for as long as the
//! `JitRhs` lives, even if the [`JitRhsBuilder`] that produced it is dropped.
//!
//! # Example
//!
//! ```
//! use tpt_sci_ode::jit::compile_rhs;
//!
//! // Compile a simple exponential decay RHS: dy/dt = -y
//! let rhs = compile_rhs(1, |_t: f64, y: &[f64], dydt: &mut [f64]| {
//!     dydt[0] = -y[0];
//! })
//! .unwrap();
//!
//! let mut dydt = [0.0];
//! rhs.call_safe(0.0, &[1.0], &mut dydt).unwrap();
//! assert!((dydt[0] + 1.0).abs() < 1e-10);
//! ```

#![allow(unsafe_code)]

use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cranelift_codegen::ir::{self, InstBuilder, types};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use target_lexicon::Triple;

use crate::error::OdeError;

/// The boxed closure type a [`JitRhs`] calls.
type RhsClosure = dyn Fn(f64, &[f64], &mut [f64]) + Send + Sync;

/// State that a compiled stub reaches through its context pointer.
struct RhsCtx {
    rhs: Box<RhsClosure>,
    /// Set by the trampoline when the closure panicked, so that the panic can
    /// be re-raised on the Rust side once the native frame has returned.
    panicked: Cell<bool>,
}

/// Rust side of the stub: runs the closure on the caller's buffers.
///
/// Panics are caught here because unwinding across the `extern "C"` boundary
/// is undefined behaviour.
extern "C" fn rhs_trampoline_extern(
    ctx: *const RhsCtx,
    nstates: usize,
    t: f64,
    y: *const f64,
    dydt: *mut f64,
) {
    // SAFETY: `ctx` is the pointer owned by the `JitRhs` that is calling this
    // stub, and it stays valid for the duration of the call.
    let ctx = unsafe { &*ctx };
    // SAFETY: `JitRhs::call` checked that both buffers hold `nstates` elements.
    let y = unsafe { std::slice::from_raw_parts(y, nstates) };
    // SAFETY: as above; `dydt` is a unique mutable buffer for this call.
    let dydt = unsafe { std::slice::from_raw_parts_mut(dydt, nstates) };
    if panic::catch_unwind(AssertUnwindSafe(|| (ctx.rhs)(t, y, dydt))).is_err() {
        ctx.panicked.set(true);
    }
}

/// A JIT-compiled RHS function for a specific state dimension.
///
/// The stub has the C signature `fn(ctx, t, y, dydt)` and is specialised to the
/// state dimension `nstates` it was compiled for. The closure it runs is owned
/// by this value and dropped with it.
pub struct JitRhs {
    /// Keeps the executable memory holding `fn_ptr` alive.
    _module: Arc<Mutex<JITModule>>,
    /// Number of state variables (dimension of y and dydt).
    nstates: usize,
    /// Entry point of the compiled stub.
    fn_ptr: *const u8,
    /// Heap-allocated closure and panic flag, owned by this value.
    ctx: *mut RhsCtx,
}

impl Drop for JitRhs {
    fn drop(&mut self) {
        // SAFETY: `ctx` came from `Box::into_raw` in `JitRhsBuilder::compile`
        // and is freed exactly once, here.
        drop(unsafe { Box::from_raw(self.ctx) });
    }
}

impl JitRhs {
    /// Returns the number of state variables this RHS operates on.
    pub fn nstates(&self) -> usize {
        self.nstates
    }

    /// Calls the compiled RHS function.
    ///
    /// # Panics
    ///
    /// Panics if the length of `y` or `dydt` does not match the compiled number
    /// of states, or if the closure itself panics. The lengths are checked in
    /// release builds too, because the stub reads and writes `nstates` elements
    /// through raw pointers. Use [`JitRhs::call_safe`] to get an error instead of
    /// a length panic.
    pub fn call(&self, t: f64, y: &[f64], dydt: &mut [f64]) {
        assert_eq!(
            y.len(),
            self.nstates,
            "y length must match compiled nstates"
        );
        assert_eq!(
            dydt.len(),
            self.nstates,
            "dydt length must match compiled nstates"
        );

        type StubFn = unsafe extern "C" fn(*const RhsCtx, f64, *const f64, *mut f64);
        // SAFETY: `fn_ptr` was produced by Cranelift for exactly this signature
        // and remains valid while `_module` is alive, which it is for `self`'s
        // lifetime.
        let stub: StubFn = unsafe { std::mem::transmute::<*const u8, StubFn>(self.fn_ptr) };
        // SAFETY: `ctx` is valid for the lifetime of `self`, and the buffer
        // lengths were checked above.
        unsafe { stub(self.ctx, t, y.as_ptr(), dydt.as_mut_ptr()) };

        // SAFETY: `ctx` is valid as above.
        let ctx = unsafe { &*self.ctx };
        if ctx.panicked.replace(false) {
            panic!("JIT-compiled RHS closure panicked");
        }
    }

    /// Calls the compiled RHS function (safe wrapper).
    ///
    /// # Errors
    ///
    /// Returns [`OdeError::Invalid`] if the length of `y` or `dydt` does not
    /// match the compiled number of states.
    pub fn call_safe(&self, t: f64, y: &[f64], dydt: &mut [f64]) -> Result<(), OdeError> {
        if y.len() != self.nstates || dydt.len() != self.nstates {
            return Err(OdeError::Invalid(format!(
                "JitRhs expects {} states, got y.len()={}, dydt.len()={}",
                self.nstates,
                y.len(),
                dydt.len()
            )));
        }
        self.call(t, y, dydt);
        Ok(())
    }
}

/// Counter that gives each compiled stub a unique symbol name.
static NEXT_STUB_ID: AtomicUsize = AtomicUsize::new(0);

/// Builder for compiling RHS functions to native code via Cranelift.
///
/// Several functions can be compiled from one builder. Each [`JitRhs`] keeps the
/// module alive, so the builder itself may be dropped after compiling.
pub struct JitRhsBuilder {
    /// The Cranelift JIT module, shared with every `JitRhs` it produces.
    module: Arc<Mutex<JITModule>>,
    /// Target ISA for code generation.
    isa: Arc<dyn TargetIsa>,
}

impl JitRhsBuilder {
    /// Creates a new JIT builder with the host's native target.
    ///
    /// # Errors
    ///
    /// Returns [`OdeError::Invalid`] if the host target ISA cannot be looked up or
    /// finalised.
    ///
    /// # Panics
    ///
    /// Panics if the internal Cranelift flag configuration is invalid (should not
    /// occur for the built-in settings).
    pub fn new() -> Result<Self, OdeError> {
        let mut flag_builder = settings::builder();
        flag_builder.enable("enable_verifier").unwrap();
        flag_builder.set("opt_level", "speed").unwrap();
        let flags = settings::Flags::new(flag_builder);

        let triple = Triple::host();
        let isa_builder = cranelift_codegen::isa::lookup(triple)
            .map_err(|e| OdeError::Invalid(format!("Failed to lookup ISA: {e}")))?;
        let isa = isa_builder
            .finish(flags)
            .map_err(|e| OdeError::Invalid(format!("Failed to create target ISA: {e}")))?;

        let mut jit_builder = JITBuilder::with_isa(isa.clone(), default_libcall_names());
        jit_builder.symbol("rhs_trampoline_extern", rhs_trampoline_extern as *const u8);
        let module = JITModule::new(jit_builder);

        Ok(Self {
            module: Arc::new(Mutex::new(module)),
            isa,
        })
    }

    /// Compiles a user-provided RHS closure to a `JitRhs`.
    ///
    /// The closure must have the signature `Fn(f64, &[f64], &mut [f64])` and
    /// will be specialised for the given `nstates` dimension.
    ///
    /// # Errors
    ///
    /// Returns [`OdeError::Invalid`] if `nstates == 0` or the stub cannot be
    /// compiled or finalised.
    pub fn compile<F>(&self, nstates: usize, rhs: F) -> Result<JitRhs, OdeError>
    where
        F: Fn(f64, &[f64], &mut [f64]) + Send + Sync + 'static,
    {
        if nstates == 0 {
            return Err(OdeError::Invalid("nstates must be > 0".to_string()));
        }

        let fn_ptr = {
            let mut module = self
                .module
                .lock()
                .map_err(|_| OdeError::Invalid("JIT module lock is poisoned".to_string()))?;
            self.define_stub(&mut module, nstates)?
        };

        let ctx = Box::into_raw(Box::new(RhsCtx {
            rhs: Box::new(rhs),
            panicked: Cell::new(false),
        }));

        Ok(JitRhs {
            _module: Arc::clone(&self.module),
            nstates,
            fn_ptr,
            ctx,
        })
    }

    /// Defines and finalises the stub `fn(ctx, t, y, dydt)` for `nstates` in
    /// `module`, returning its entry point.
    fn define_stub(&self, module: &mut JITModule, nstates: usize) -> Result<*const u8, OdeError> {
        let call_conv = CallConv::for_libcall(
            self.isa.flags(),
            CallConv::triple_default(self.isa.triple()),
        );

        // Signature of the Rust trampoline: (ctx, nstates, t, y, dydt).
        let mut tramp_sig = ir::Signature::new(call_conv);
        tramp_sig.params.push(ir::AbiParam::new(types::I64)); // ctx
        tramp_sig.params.push(ir::AbiParam::new(types::I64)); // nstates
        tramp_sig.params.push(ir::AbiParam::new(types::F64)); // t
        tramp_sig.params.push(ir::AbiParam::new(types::I64)); // y
        tramp_sig.params.push(ir::AbiParam::new(types::I64)); // dydt
        let tramp_id = module
            .declare_function("rhs_trampoline_extern", Linkage::Import, &tramp_sig)
            .map_err(|e| OdeError::Invalid(format!("Failed to declare trampoline: {e}")))?;

        // Signature of the stub: (ctx, t, y, dydt).
        let mut sig = ir::Signature::new(call_conv);
        sig.params.push(ir::AbiParam::new(types::I64)); // ctx
        sig.params.push(ir::AbiParam::new(types::F64)); // t
        sig.params.push(ir::AbiParam::new(types::I64)); // y
        sig.params.push(ir::AbiParam::new(types::I64)); // dydt

        let id = NEXT_STUB_ID.fetch_add(1, Ordering::Relaxed);
        let func_id = module
            .declare_function(&format!("jit_rhs_{id}"), Linkage::Export, &sig)
            .map_err(|e| OdeError::Invalid(format!("Failed to declare function: {e}")))?;

        let mut cl_ctx = module.make_context();
        cl_ctx.func.signature = sig;
        {
            let mut func_ctx = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut cl_ctx.func, &mut func_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let ctx_val = builder.block_params(entry)[0];
            let t_val = builder.block_params(entry)[1];
            let y_ptr = builder.block_params(entry)[2];
            let dydt_ptr = builder.block_params(entry)[3];

            let tramp_ref = module.declare_func_in_func(tramp_id, builder.func);
            let n_val = builder.ins().iconst(types::I64, nstates as i64);
            builder
                .ins()
                .call(tramp_ref, &[ctx_val, n_val, t_val, y_ptr, dydt_ptr]);
            builder.ins().return_(&[]);
            builder.finalize(self.isa.frontend_config());
        }

        module
            .define_function(func_id, &mut cl_ctx)
            .map_err(|e| OdeError::Invalid(format!("Failed to define function: {e}")))?;
        module
            .finalize_definitions()
            .map_err(|e| OdeError::Invalid(format!("Failed to finalize definitions: {e}")))?;

        Ok(module.get_finalized_function(func_id))
    }
}

/// Compiles a user RHS closure to a JIT-compiled call stub.
///
/// This is a convenience function that creates a builder and compiles the closure
/// in one step. For repeated compilations, use `JitRhsBuilder` directly.
///
/// # Errors
///
/// Returns [`OdeError::Invalid`] if `nstates == 0` or compilation fails (see
/// [`JitRhsBuilder::compile`]).
///
/// # Example
///
/// ```
/// use tpt_sci_ode::jit::compile_rhs;
///
/// // Lotka-Volterra equations
/// let rhs = compile_rhs(2, |_t: f64, y: &[f64], dydt: &mut [f64]| {
///     dydt[0] = 1.5 * y[0] - 1.0 * y[0] * y[1];
///     dydt[1] = -3.0 * y[1] + 1.0 * y[0] * y[1];
/// })
/// .unwrap();
/// assert_eq!(rhs.nstates(), 2);
/// ```
pub fn compile_rhs<F>(nstates: usize, rhs: F) -> Result<JitRhs, OdeError>
where
    F: Fn(f64, &[f64], &mut [f64]) + Send + Sync + 'static,
{
    JitRhsBuilder::new()?.compile(nstates, rhs)
}

impl Default for JitRhsBuilder {
    fn default() -> Self {
        Self::new().expect("Failed to create JitRhsBuilder")
    }
}

/// `JitRhs` is a [`RhsCallable`](crate::RhsCallable) so it can be fed directly
/// into [`crate::OdeProblem`] / the integrators, sharing the exact same solving
/// pipeline as a plain closure.
impl crate::RhsCallable for JitRhs {
    fn nstates(&self) -> usize {
        self.nstates
    }

    fn call(&self, t: f64, y: &[f64], dydt: &mut [f64]) -> Result<(), OdeError> {
        self.call_safe(t, y, dydt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_builder_creation() {
        assert!(JitRhsBuilder::new().is_ok());
    }

    #[test]
    fn test_compile_simple_rhs() {
        let builder = JitRhsBuilder::new().unwrap();
        let jit_rhs = builder
            .compile(1, |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            })
            .unwrap();

        assert_eq!(jit_rhs.nstates(), 1);
        let mut dydt = [0.0];
        jit_rhs.call_safe(0.0, &[1.0], &mut dydt).unwrap();
        assert!((dydt[0] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compile_lotka_volterra() {
        let builder = JitRhsBuilder::new().unwrap();
        let jit_rhs = builder
            .compile(2, |_t: f64, y: &[f64], dydt: &mut [f64]| {
                let alpha = 1.5;
                let beta = 1.0;
                let gamma = 3.0;
                let delta = 1.0;
                dydt[0] = alpha * y[0] - beta * y[0] * y[1];
                dydt[1] = delta * y[0] * y[1] - gamma * y[1];
            })
            .unwrap();

        assert_eq!(jit_rhs.nstates(), 2);
        let mut dydt = [0.0, 0.0];
        jit_rhs.call_safe(0.0, &[10.0, 5.0], &mut dydt).unwrap();

        // dy0/dt = 1.5*10 - 1.0*10*5 = -35;  dy1/dt = 1.0*10*5 - 3.0*5 = 35
        assert!((dydt[0] + 35.0).abs() < 1e-10);
        assert!((dydt[1] - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_compile_rhs_convenience() {
        let jit_rhs = compile_rhs(1, |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -2.0 * y[0];
        })
        .unwrap();

        assert_eq!(jit_rhs.nstates(), 1);
        let mut dydt = [0.0];
        jit_rhs.call_safe(0.0, &[1.0], &mut dydt).unwrap();
        assert!((dydt[0] + 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_function_outlives_builder() {
        // Regression: the module that owns the machine code used to be dropped
        // with the builder, leaving a dangling function pointer.
        let jit_rhs = {
            let builder = JitRhsBuilder::new().unwrap();
            builder
                .compile(1, |_t: f64, y: &[f64], dydt: &mut [f64]| {
                    dydt[0] = -3.0 * y[0];
                })
                .unwrap()
        };
        let mut dydt = [0.0];
        jit_rhs.call_safe(0.0, &[2.0], &mut dydt).unwrap();
        assert!((dydt[0] + 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_functions_from_one_builder() {
        let builder = JitRhsBuilder::new().unwrap();
        let a = builder
            .compile(1, |_t: f64, y: &[f64], dydt: &mut [f64]| dydt[0] = y[0])
            .unwrap();
        let b = builder
            .compile(1, |_t: f64, y: &[f64], dydt: &mut [f64]| dydt[0] = -y[0])
            .unwrap();
        let mut da = [0.0];
        let mut db = [0.0];
        a.call_safe(0.0, &[1.0], &mut da).unwrap();
        b.call_safe(0.0, &[1.0], &mut db).unwrap();
        assert_eq!(da[0], 1.0);
        assert_eq!(db[0], -1.0);
    }

    #[test]
    #[should_panic(expected = "y length must match")]
    fn test_call_panics_on_length_mismatch() {
        let jit_rhs = compile_rhs(1, |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -y[0];
        })
        .unwrap();
        let mut dydt = [0.0];
        jit_rhs.call(0.0, &[1.0, 2.0], &mut dydt);
    }

    #[test]
    #[should_panic(expected = "JIT-compiled RHS closure panicked")]
    fn test_closure_panic_is_reraised_on_rust_side() {
        let jit_rhs = compile_rhs(1, |_t: f64, _y: &[f64], _dydt: &mut [f64]| {
            panic!("boom");
        })
        .unwrap();
        let mut dydt = [0.0];
        jit_rhs.call(0.0, &[1.0], &mut dydt);
    }

    #[test]
    fn test_rhs_callable_trait() {
        let closure_rhs = |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -y[0];
        };
        let mut dydt = [0.0];
        closure_rhs(0.0, &[1.0], &mut dydt);
        assert!((dydt[0] + 1.0).abs() < 1e-10);
    }
}
