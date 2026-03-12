//! Utilities to construct pliron IR and execute it.

use std::cell::RefCell;

use burn_tensor::{DType, TensorData};
use pliron::{
    builtin::{op_interfaces::SingleBlockRegionInterface, types::Signedness},
    context::Context,
    irbuild::{
        dialect_conversion::apply_dialect_conversion,
        inserter::{IRInserter, Inserter},
        listener::DummyListener,
    },
    op::{Op, verify_op},
    result::ExpectOk,
    r#type::TypePtr,
};
use pliron_common_dialects::cf::to_llvm::CFToLLVM;
use pliron_llvm::llvm_sys::{core::LLVMContext, lljit::LLVMLLJIT, target::initialize_native};
use pliron_tensor::{
    memref::{conversions::MemrefToCF, type_interfaces::Dimension},
    tensor::{
        conversions::TensorToMemref, runtime_utils::TensorDesciptor, types::RankedTensorType,
    },
};

#[derive(Default)]
pub struct PlironIR {
    /// Pliron context that owns everything Pliron related.
    pub ctx: Context,
    /// Convenience for appending the next IR instruction.
    pub inserter: IRInserter<DummyListener>,
    /// Owned tensors
    pub tensors: Vec<TensorData>,
    /// IR Tensor desciptors that needs to be live throughout.
    pub tensor_descriptors: Vec<Vec<u8>>,
    /// The IR module that we are building, for convenience.
    pub module: Option<pliron::builtin::ops::ModuleOp>,
}

thread_local! {
    pub static PLIRON_IR: RefCell<Option<PlironIR>> = RefCell::new(None);
}

/// Convert a burn tensor data to a pliron tensor descriptor,
/// which can be used to construct pliron tensors.
/// The caller must ensure that the tensor data is not dropped
/// while the pliron tensor is still in use by the IR.
pub(crate) fn to_pliron_tensor_descriptor(t: &burn_tensor::TensorData) -> TensorDesciptor {
    TensorDesciptor::new(t.shape.clone(), t.dtype.size(), t.bytes.as_ptr())
}

/// Initialize (if not already) [PlironIR] for the current thread.
pub(crate) fn init_pliron_ir() {
    if is_pliron_ir_initialized() {
        return;
    }
    PLIRON_IR.with(|ir| {
        if ir.borrow().is_none() {
            *ir.borrow_mut() = Some(PlironIR::default());
        }
        let mut ir = ir.borrow_mut();
        let ir = ir.as_mut().unwrap();
        // Insert a module, and a function inside it, so that we can insert instructions later.
        let module =
            pliron::builtin::ops::ModuleOp::new(&mut ir.ctx, "burn_exec".try_into().unwrap());
        let void_ty = pliron_llvm::types::VoidType::get(&ir.ctx);
        let func_ty = pliron_llvm::types::FuncType::get(&mut ir.ctx, void_ty.into(), vec![], false);
        let func = pliron_llvm::ops::FuncOp::new(&mut ir.ctx, "main".try_into().unwrap(), func_ty);
        func.get_operation()
            .insert_at_back(module.get_body(&mut ir.ctx, 0), &mut ir.ctx);
        let entry_block = func.get_or_create_entry_block(&mut ir.ctx);
        // Set the insertion point to the end of the entry block, so that we can insert instructions there.
        ir.inserter.set_insertion_point_to_block_end(entry_block);
        ir.module = Some(module);

        // Initialise the native target for LLVM execution.
        initialize_native().expect("Failed to initialize native target for LLVM execution");
    })
}

/// Is [PlironIR] initialized for the current thread?
pub(crate) fn is_pliron_ir_initialized() -> bool {
    PLIRON_IR.with(|ir| ir.borrow().is_some())
}

/// Clear the pliron IR for the current thread. This should be called after executing the IR
/// to free the resources.
pub(crate) fn clear_pliron_ir() {
    PLIRON_IR.with(|ir| {
        *ir.borrow_mut() = None;
    });
}

pub(crate) fn build_ranked_tensor_type(
    ctx: &mut Context,
    burn_ty: DType,
    shape: &[usize],
) -> TypePtr<RankedTensorType> {
    let element_type = match burn_ty {
        DType::F32 => pliron::builtin::types::FP32Type::get(ctx).into(),
        DType::F64 => pliron::builtin::types::FP64Type::get(ctx).into(),
        DType::U64 => {
            pliron::builtin::types::IntegerType::get(ctx, 64, Signedness::Signless).into()
        }
        DType::Bool => {
            pliron::builtin::types::IntegerType::get(ctx, 1, Signedness::Signless).into()
        }
        _ => unimplemented!("Unsupported data type"),
    };
    RankedTensorType::get(
        ctx,
        element_type,
        shape.iter().map(|&dim| Dimension::Static(dim)).collect(),
    )
}

pub(crate) fn exec_pliron_ir(ir: &mut PlironIR) {
    let module = ir.module.unwrap();
    let module_operation = module.get_operation();
    apply_dialect_conversion(&mut ir.ctx, &mut TensorToMemref, module_operation).expect_ok(&ir.ctx);
    apply_dialect_conversion(&mut ir.ctx, &mut MemrefToCF, module_operation).expect_ok(&ir.ctx);
    apply_dialect_conversion(&mut ir.ctx, &mut CFToLLVM, module_operation).expect_ok(&ir.ctx);
    verify_op(&module, &ir.ctx).expect_ok(&ir.ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir =
        pliron_llvm::to_llvm_ir::convert_module(&ir.ctx, &llvm_ctx, module).expect_ok(&ir.ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit.lookup_symbol("main").expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let f = unsafe { std::mem::transmute::<u64, extern "C" fn() -> ()>(symbol_addr) };
    f();
}
