//! Tensor ops implementation for Pliron backend
#![allow(unused_variables)]

use std::num::NonZero;

use burn_backend::{
    TensorMetadata,
    ops::{
        ActivationOps, BoolTensorOps, FloatTensorOps, IntTensorOps, ModuleOps, QTensorOps,
        TransactionOps,
    },
};
use burn_std::{Shape, TensorData};
use pliron::{
    builtin::{
        attributes::{FPSingleAttr, IntegerAttr},
        op_interfaces::OneResultInterface,
        types::{IntegerType, Signedness},
    },
    irbuild::inserter::Inserter,
    utils::apint::APInt,
};
use pliron_llvm::op_interfaces::CastOpInterface;
use pliron_tensor::tensor::runtime_utils::TensorDesciptor;

use crate::{
    Pliron, PlironBoolTensor, PlironDevice, PlironFloatTensor, PlironIntTensor, PlironQTensor,
    ir_interface::{
        PLIRON_IR, build_ranked_tensor_type, clear_pliron_ir, exec_pliron_ir, init_pliron_ir,
        is_pliron_ir_initialized, to_pliron_tensor_descriptor,
    },
};

impl<F: Send + Sync + 'static, I: Send + Sync + 'static, B: Send + Sync + 'static>
    FloatTensorOps<Pliron<F, I, B>> for Pliron<F, I, B>
{
    fn float_from_data(data: TensorData, device: &PlironDevice) -> PlironFloatTensor {
        init_pliron_ir();
        PLIRON_IR.with_borrow_mut(|ir| {
            let ir = ir.as_mut().unwrap();
            // Save the tensor data to the IR interface so it lives long
            // enough for the pliron value to reference it.
            let pliron_tensor_descriptor = to_pliron_tensor_descriptor(&data);
            let tensor_ty = build_ranked_tensor_type(&mut ir.ctx, data.dtype, &data.shape);
            let shape = data.shape.clone();
            ir.tensors.push(data);
            // The pointer in pliron_tensor_descriptor is stored as an i64 constant in pliron
            // and converted to a pointer in the IR, and then loaded as a tensor value.
            let tensor_data_addr = pliron_tensor_descriptor.build_ir_descriptor();
            let i64_ty = IntegerType::get(&mut ir.ctx, 64, Signedness::Signless);
            let addr_attr_val = APInt::from_u64(
                tensor_data_addr.as_ptr() as u64,
                NonZero::<usize>::new(64).unwrap(),
            );
            // We need the tensor descriptor to be live until the end of the IR execution.
            ir.tensor_descriptors.push(tensor_data_addr);
            let addr_attr = IntegerAttr::new(i64_ty, addr_attr_val);
            let const_op = pliron_llvm::ops::ConstantOp::new(&mut ir.ctx, Box::new(addr_attr));
            ir.inserter.insert_op(&ir.ctx, &const_op);
            let ptr_ty = pliron_llvm::types::PointerType::get(&mut ir.ctx);
            let const_addr = const_op.get_result(&ir.ctx);
            let int_to_ptr =
                pliron_llvm::ops::IntToPtrOp::new(&mut ir.ctx, const_addr, ptr_ty.into());
            ir.inserter.insert_op(&mut ir.ctx, &int_to_ptr);
            // Load the tensor data as a pliron value. The load op will have the tensor type as its result type,
            // and the pointer to the tensor data as its operand.
            let ptr_val = int_to_ptr.get_result(&ir.ctx);
            let load_op = pliron_llvm::ops::LoadOp::new(&mut ir.ctx, ptr_val, tensor_ty.into());
            ir.inserter.insert_op(&mut ir.ctx, &load_op);
            PlironFloatTensor {
                shape: shape.into(),
                value: load_op.get_result(&ir.ctx),
            }
        })
    }

    fn float_random(
        shape: Shape,
        distribution: burn_std::Distribution,
        device: &PlironDevice,
        dtype: burn_std::FloatDType,
    ) -> PlironFloatTensor {
        todo!()
    }

    async fn float_into_data(
        tensor: PlironFloatTensor,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        assert!(
            is_pliron_ir_initialized(),
            "Pliron IR must be initialized to convert a pliron tensor into data. \
            This usually means that the tensor is not created from data, or the IR \
            is cleared after execution."
        );
        let result = PLIRON_IR.with_borrow_mut(|ir| {
            let ir = ir.as_mut().unwrap();
            // Create a tensor descriptor for the given pliron tensor, and
            // add IR code to store to it.
            let elemement_size = tensor.dtype().size();
            let shape = tensor.shape();
            let rank = shape.rank();

            let tensor_descriptor =
                TensorDesciptor::new(shape.clone().into(), elemement_size, std::ptr::null());
            let mut ir_descriptor = tensor_descriptor.build_ir_descriptor();
            let ir_descriptor_addr = ir_descriptor.as_mut_ptr();
            // We need the tensor descriptor to be live until the end of the IR execution.
            ir.tensor_descriptors.push(ir_descriptor);

            // Create a constant integer and convert it to a pointer
            let i64_ty = IntegerType::get(&mut ir.ctx, 64, Signedness::Signless);
            let addr_attr_val = APInt::from_u64(
                ir_descriptor_addr as u64,
                NonZero::<usize>::new(64).unwrap(),
            );
            let addr_attr = IntegerAttr::new(i64_ty, addr_attr_val);
            let const_op = pliron_llvm::ops::ConstantOp::new(&mut ir.ctx, Box::new(addr_attr));
            ir.inserter.insert_op(&mut ir.ctx, &const_op);
            let ptr_ty = pliron_llvm::types::PointerType::get(&mut ir.ctx);
            let const_addr = const_op.get_result(&ir.ctx);
            let int_to_ptr =
                pliron_llvm::ops::IntToPtrOp::new(&mut ir.ctx, const_addr, ptr_ty.into());
            ir.inserter.insert_op(&mut ir.ctx, &int_to_ptr);
            let ptr_val = int_to_ptr.get_result(&mut ir.ctx);
            // Store the tensor data to the address of the tensor descriptor.
            let store_op = pliron_llvm::ops::StoreOp::new(&mut ir.ctx, tensor.value, ptr_val);
            ir.inserter.insert_op(&mut ir.ctx, &store_op);

            // Insert a return to complete the function, so that the IR can be executed
            // and the tensor data can be read back from the tensor descriptor.
            let ret = pliron_llvm::ops::ReturnOp::new(&mut ir.ctx, None);
            ir.inserter.insert_op(&mut ir.ctx, &ret);

            // Execute the IR.
            exec_pliron_ir(ir);

            let descr = unsafe {
                TensorDesciptor::from_ir_descriptor(ir_descriptor_addr, rank, elemement_size)
            };
            let data_ptr = unsafe {
                std::slice::from_raw_parts(descr.aligned_ptr(), descr.total_size_in_bytes())
            };
            Ok(TensorData::from_bytes_vec(
                data_ptr.to_vec(),
                shape,
                tensor.dtype(),
            ))
        });
        clear_pliron_ir();
        result
    }

    fn float_device(tensor: &PlironFloatTensor) -> PlironDevice {
        PlironDevice
    }

    fn float_to_device(tensor: PlironFloatTensor, device: &PlironDevice) -> PlironFloatTensor {
        todo!()
    }

    fn float_into_int(tensor: PlironFloatTensor, out_dtype: burn_std::IntDType) -> PlironIntTensor {
        todo!()
    }

    fn float_empty(
        shape: Shape,
        device: &PlironDevice,
        dtype: burn_std::FloatDType,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn float_add(lhs: PlironFloatTensor, rhs: PlironFloatTensor) -> PlironFloatTensor {
        assert!(is_pliron_ir_initialized(),);
        PLIRON_IR.with_borrow_mut(|ir| {
            let ir = ir.as_mut().unwrap();
            let sum = pliron_tensor::tensor::ops::AddOp::new(&mut ir.ctx, lhs.value, rhs.value);
            ir.inserter.insert_op(&mut ir.ctx, &sum);
            PlironFloatTensor {
                shape: lhs.shape.clone(),
                value: sum.get_result(&ir.ctx),
            }
        })
    }

    fn float_add_scalar(lhs: PlironFloatTensor, rhs: burn_std::Scalar) -> PlironFloatTensor {
        todo!()
    }

    fn float_sub(lhs: PlironFloatTensor, rhs: PlironFloatTensor) -> PlironFloatTensor {
        assert!(is_pliron_ir_initialized());
        PLIRON_IR.with_borrow_mut(|ir| {
            let ir = ir.as_mut().unwrap();
            let diff = pliron_tensor::tensor::ops::SubOp::new(&mut ir.ctx, lhs.value, rhs.value);
            ir.inserter.insert_op(&mut ir.ctx, &diff);
            PlironFloatTensor {
                shape: lhs.shape.clone(),
                value: diff.get_result(&ir.ctx),
            }
        })
    }

    fn float_sub_scalar(lhs: PlironFloatTensor, rhs: burn_std::Scalar) -> PlironFloatTensor {
        todo!()
    }

    fn float_mul(lhs: PlironFloatTensor, rhs: PlironFloatTensor) -> PlironFloatTensor {
        assert!(is_pliron_ir_initialized());
        PLIRON_IR.with_borrow_mut(|ir| {
            let ir = ir.as_mut().unwrap();
            let prod = pliron_tensor::tensor::ops::MulOp::new(&mut ir.ctx, lhs.value, rhs.value);
            ir.inserter.insert_op(&mut ir.ctx, &prod);
            PlironFloatTensor {
                shape: lhs.shape.clone(),
                value: prod.get_result(&ir.ctx),
            }
        })
    }

    fn float_mul_scalar(lhs: PlironFloatTensor, rhs: burn_std::Scalar) -> PlironFloatTensor {
        todo!()
    }

    fn float_div(lhs: PlironFloatTensor, rhs: PlironFloatTensor) -> PlironFloatTensor {
        assert!(is_pliron_ir_initialized());
        PLIRON_IR.with_borrow_mut(|ir| {
            let ir = ir.as_mut().unwrap();
            let quot = pliron_tensor::tensor::ops::DivOp::new(&mut ir.ctx, lhs.value, rhs.value);
            ir.inserter.insert_op(&mut ir.ctx, &quot);
            PlironFloatTensor {
                shape: lhs.shape.clone(),
                value: quot.get_result(&ir.ctx),
            }
        })
    }

    fn float_div_scalar(lhs: PlironFloatTensor, rhs: burn_std::Scalar) -> PlironFloatTensor {
        todo!()
    }

    fn float_remainder(lhs: PlironFloatTensor, rhs: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_remainder_scalar(lhs: PlironFloatTensor, rhs: burn_std::Scalar) -> PlironFloatTensor {
        todo!()
    }

    fn float_matmul(lhs: PlironFloatTensor, rhs: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_cross(
        lhs: PlironFloatTensor,
        rhs: PlironFloatTensor,
        dim: usize,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn float_recip(tensor: PlironFloatTensor) -> PlironFloatTensor {
        assert!(is_pliron_ir_initialized());
        PLIRON_IR.with_borrow_mut(|ir| {
            let ir = ir.as_mut().unwrap();
            let tensor_ty = build_ranked_tensor_type(&mut ir.ctx, tensor.dtype(), &tensor.shape);
            let generate_ones = pliron_tensor::tensor::ops::GenerateOp::new(
                &mut ir.ctx,
                vec![],
                tensor_ty,
                |ctx, _state, inserter, _indices| {
                    let one_attr = FPSingleAttr::from(1.0);
                    let return_one = pliron_llvm::ops::ConstantOp::new(ctx, Box::new(one_attr));
                    inserter.insert_op(ctx, &return_one);
                    return_one.get_result(ctx)
                },
                (),
            );
            ir.inserter.insert_op(&ir.ctx, &generate_ones);
            let ones = generate_ones.get_result(&ir.ctx);
            let recip = pliron_tensor::tensor::ops::DivOp::new(&mut ir.ctx, ones, tensor.value);
            ir.inserter.insert_op(&ir.ctx, &recip);
            PlironFloatTensor {
                shape: tensor.shape.clone(),
                value: recip.get_result(&ir.ctx),
            }
        })
    }

    fn float_swap_dims(tensor: PlironFloatTensor, dim1: usize, dim2: usize) -> PlironFloatTensor {
        todo!()
    }

    fn float_permute(tensor: PlironFloatTensor, axes: &[usize]) -> PlironFloatTensor {
        todo!()
    }

    fn float_flip(tensor: PlironFloatTensor, axes: &[usize]) -> PlironFloatTensor {
        todo!()
    }

    fn float_reshape(tensor: PlironFloatTensor, shape: Shape) -> PlironFloatTensor {
        todo!()
    }

    fn float_gather(
        dim: usize,
        tensor: PlironFloatTensor,
        indices: PlironIntTensor,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn float_scatter_add(
        dim: usize,
        tensor: PlironFloatTensor,
        indices: PlironIntTensor,
        value: PlironFloatTensor,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn float_select(
        tensor: PlironFloatTensor,
        dim: usize,
        indices: PlironIntTensor,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn float_select_add(
        tensor: PlironFloatTensor,
        dim: usize,
        indices: PlironIntTensor,
        value: PlironFloatTensor,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn float_slice(tensor: PlironFloatTensor, slices: &[burn_std::Slice]) -> PlironFloatTensor {
        todo!()
    }

    fn float_slice_assign(
        tensor: PlironFloatTensor,
        slices: &[burn_std::Slice],
        value: PlironFloatTensor,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn float_mask_where(
        tensor: PlironFloatTensor,
        mask: PlironBoolTensor,
        value: PlironFloatTensor,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn float_mask_fill(
        tensor: PlironFloatTensor,
        mask: PlironBoolTensor,
        value: burn_std::Scalar,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn float_equal(
        lhs: PlironFloatTensor,
        rhs: PlironFloatTensor,
        out_dtype: burn_std::BoolDType,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn float_equal_elem(
        lhs: PlironFloatTensor,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn float_greater(
        lhs: PlironFloatTensor,
        rhs: PlironFloatTensor,
        out_dtype: burn_std::BoolDType,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn float_greater_elem(
        lhs: PlironFloatTensor,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn float_greater_equal(
        lhs: PlironFloatTensor,
        rhs: PlironFloatTensor,
        out_dtype: burn_std::BoolDType,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn float_greater_equal_elem(
        lhs: PlironFloatTensor,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn float_lower(
        lhs: PlironFloatTensor,
        rhs: PlironFloatTensor,
        out_dtype: burn_std::BoolDType,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn float_lower_elem(
        lhs: PlironFloatTensor,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn float_lower_equal(
        lhs: PlironFloatTensor,
        rhs: PlironFloatTensor,
        out_dtype: burn_std::BoolDType,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn float_lower_equal_elem(
        lhs: PlironFloatTensor,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn float_sum(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_sum_dim(tensor: PlironFloatTensor, dim: usize) -> PlironFloatTensor {
        todo!()
    }

    fn float_mean_dim(tensor: PlironFloatTensor, dim: usize) -> PlironFloatTensor {
        todo!()
    }

    fn float_cumsum(tensor: PlironFloatTensor, dim: usize) -> PlironFloatTensor {
        todo!()
    }

    fn float_cumprod(tensor: PlironFloatTensor, dim: usize) -> PlironFloatTensor {
        todo!()
    }

    fn float_cummin(tensor: PlironFloatTensor, dim: usize) -> PlironFloatTensor {
        todo!()
    }

    fn float_cummax(tensor: PlironFloatTensor, dim: usize) -> PlironFloatTensor {
        todo!()
    }

    fn float_cast(tensor: PlironFloatTensor, dtype: burn_std::FloatDType) -> PlironFloatTensor {
        todo!()
    }

    fn float_exp(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_log(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_log1p(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_powf(lhs: PlironFloatTensor, rhs: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_powf_scalar_impl(
        tensor: PlironFloatTensor,
        value: burn_std::Scalar,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn float_sqrt(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_abs(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_cos(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_sin(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_tan(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_cosh(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_sinh(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_tanh(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_acos(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_acosh(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_asin(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_asinh(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_atan(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_atanh(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_atan2(lhs: PlironFloatTensor, rhs: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_round(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_floor(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_ceil(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_trunc(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_erf(tensor: PlironFloatTensor) -> PlironFloatTensor {
        todo!()
    }

    fn float_argmax(
        tensor: PlironFloatTensor,
        dim: usize,
        out_dtype: burn_std::IntDType,
    ) -> PlironIntTensor {
        todo!()
    }

    fn float_argmin(
        tensor: PlironFloatTensor,
        dim: usize,
        out_dtype: burn_std::IntDType,
    ) -> PlironIntTensor {
        todo!()
    }

    fn float_argtopk(
        tensor: PlironFloatTensor,
        dim: usize,
        k: usize,
        out_dtype: burn_std::IntDType,
    ) -> PlironIntTensor {
        todo!()
    }

    fn float_expand(tensor: PlironFloatTensor, shape: Shape) -> PlironFloatTensor {
        todo!()
    }

    fn float_unfold(
        tensor: PlironFloatTensor,
        dim: usize,
        size: usize,
        step: usize,
    ) -> PlironFloatTensor {
        todo!()
    }
}

impl<F: Send + Sync + 'static, I: Send + Sync + 'static, B: Send + Sync + 'static>
    BoolTensorOps<Pliron<F, I, B>> for Pliron<F, I, B>
{
    fn bool_empty(
        shape: Shape,
        device: &PlironDevice,
        bool_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn bool_zeros(
        shape: Shape,
        device: &PlironDevice,
        bool_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn bool_ones(
        shape: Shape,
        device: &PlironDevice,
        bool_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    async fn bool_into_data(
        tensor: PlironBoolTensor,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        todo!()
    }

    fn bool_from_data(data: TensorData, device: &PlironDevice) -> PlironBoolTensor {
        todo!()
    }

    fn bool_into_int(tensor: PlironBoolTensor, out_dtype: burn_std::IntDType) -> PlironIntTensor {
        todo!()
    }

    fn bool_into_float(
        tensor: PlironBoolTensor,
        out_dtype: burn_std::FloatDType,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn bool_select(
        tensor: PlironBoolTensor,
        dim: usize,
        indices: PlironIntTensor,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn bool_select_or(
        tensor: PlironBoolTensor,
        dim: usize,
        indices: PlironIntTensor,
        value: PlironBoolTensor,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn bool_device(tensor: &PlironBoolTensor) -> PlironDevice {
        todo!()
    }

    fn bool_to_device(tensor: PlironBoolTensor, device: &PlironDevice) -> PlironBoolTensor {
        todo!()
    }

    fn bool_reshape(tensor: PlironBoolTensor, shape: Shape) -> PlironBoolTensor {
        todo!()
    }

    fn bool_slice(tensor: PlironBoolTensor, slices: &[burn_std::Slice]) -> PlironBoolTensor {
        todo!()
    }

    fn bool_slice_assign(
        tensor: PlironBoolTensor,
        slices: &[burn_std::Slice],
        value: PlironBoolTensor,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn bool_mask_where(
        tensor: PlironBoolTensor,
        mask: PlironBoolTensor,
        value: PlironBoolTensor,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn bool_mask_fill(
        tensor: PlironBoolTensor,
        mask: PlironBoolTensor,
        value: burn_std::Scalar,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn bool_gather(
        dim: usize,
        tensor: PlironBoolTensor,
        indices: PlironIntTensor,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: PlironBoolTensor,
        indices: PlironIntTensor,
        value: PlironBoolTensor,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn bool_equal(lhs: PlironBoolTensor, rhs: PlironBoolTensor) -> PlironBoolTensor {
        todo!()
    }

    fn bool_equal_elem(lhs: PlironBoolTensor, rhs: burn_std::Scalar) -> PlironBoolTensor {
        todo!()
    }

    fn bool_not(tensor: PlironBoolTensor) -> PlironBoolTensor {
        todo!()
    }

    fn bool_and(tensor: PlironBoolTensor, rhs: PlironBoolTensor) -> PlironBoolTensor {
        todo!()
    }

    fn bool_or(tensor: PlironBoolTensor, rhs: PlironBoolTensor) -> PlironBoolTensor {
        todo!()
    }

    fn bool_swap_dims(tensor: PlironBoolTensor, dim1: usize, dim2: usize) -> PlironBoolTensor {
        todo!()
    }

    fn bool_permute(tensor: PlironBoolTensor, axes: &[usize]) -> PlironBoolTensor {
        todo!()
    }

    fn bool_flip(tensor: PlironBoolTensor, axes: &[usize]) -> PlironBoolTensor {
        todo!()
    }

    fn bool_expand(tensor: PlironBoolTensor, shape: Shape) -> PlironBoolTensor {
        todo!()
    }

    fn bool_unfold(
        tensor: PlironBoolTensor,
        dim: usize,
        size: usize,
        step: usize,
    ) -> PlironBoolTensor {
        todo!()
    }
}

impl<F: Send + Sync + 'static, I: Send + Sync + 'static, B: Send + Sync + 'static>
    IntTensorOps<Pliron<F, I, B>> for Pliron<F, I, B>
{
    fn int_empty(
        shape: Shape,
        device: &PlironDevice,
        dtype: burn_std::IntDType,
    ) -> PlironIntTensor {
        todo!()
    }

    async fn int_into_data(
        tensor: PlironIntTensor,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        todo!()
    }

    fn int_from_data(data: TensorData, device: &PlironDevice) -> PlironIntTensor {
        todo!()
    }

    fn int_device(tensor: &PlironIntTensor) -> PlironDevice {
        todo!()
    }

    fn int_to_device(tensor: PlironIntTensor, device: &PlironDevice) -> PlironIntTensor {
        todo!()
    }

    fn int_reshape(tensor: PlironIntTensor, shape: Shape) -> PlironIntTensor {
        todo!()
    }

    fn int_slice(tensor: PlironIntTensor, slices: &[burn_std::Slice]) -> PlironIntTensor {
        todo!()
    }

    fn int_slice_assign(
        tensor: PlironIntTensor,
        slices: &[burn_std::Slice],
        value: PlironIntTensor,
    ) -> PlironIntTensor {
        todo!()
    }

    fn int_into_float(
        tensor: PlironIntTensor,
        out_dtype: burn_std::FloatDType,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn int_mask_where(
        tensor: PlironIntTensor,
        mask: PlironBoolTensor,
        value: PlironIntTensor,
    ) -> PlironIntTensor {
        todo!()
    }

    fn int_mask_fill(
        tensor: PlironIntTensor,
        mask: PlironBoolTensor,
        value: burn_std::Scalar,
    ) -> PlironIntTensor {
        todo!()
    }

    fn int_gather(
        dim: usize,
        tensor: PlironIntTensor,
        indices: PlironIntTensor,
    ) -> PlironIntTensor {
        todo!()
    }

    fn int_scatter_add(
        dim: usize,
        tensor: PlironIntTensor,
        indices: PlironIntTensor,
        value: PlironIntTensor,
    ) -> PlironIntTensor {
        todo!()
    }

    fn int_select(
        tensor: PlironIntTensor,
        dim: usize,
        indices: PlironIntTensor,
    ) -> PlironIntTensor {
        todo!()
    }

    fn int_select_add(
        tensor: PlironIntTensor,
        dim: usize,
        indices: PlironIntTensor,
        value: PlironIntTensor,
    ) -> PlironIntTensor {
        todo!()
    }

    fn int_equal(
        lhs: PlironIntTensor,
        rhs: PlironIntTensor,
        out_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn int_equal_elem(
        lhs: PlironIntTensor,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn int_greater(
        lhs: PlironIntTensor,
        rhs: PlironIntTensor,
        out_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn int_greater_elem(
        lhs: PlironIntTensor,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn int_greater_equal(
        lhs: PlironIntTensor,
        rhs: PlironIntTensor,
        out_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn int_greater_equal_elem(
        lhs: PlironIntTensor,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn int_lower(
        lhs: PlironIntTensor,
        rhs: PlironIntTensor,
        out_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn int_lower_elem(
        lhs: PlironIntTensor,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn int_lower_equal(
        lhs: PlironIntTensor,
        rhs: PlironIntTensor,
        out_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn int_lower_equal_elem(
        lhs: PlironIntTensor,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolStore,
    ) -> PlironBoolTensor {
        todo!()
    }

    fn int_add(lhs: PlironIntTensor, rhs: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn int_add_scalar(lhs: PlironIntTensor, rhs: burn_std::Scalar) -> PlironIntTensor {
        todo!()
    }

    fn int_sub(lhs: PlironIntTensor, rhs: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn int_sub_scalar(lhs: PlironIntTensor, rhs: burn_std::Scalar) -> PlironIntTensor {
        todo!()
    }

    fn int_mul(lhs: PlironIntTensor, rhs: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn int_mul_scalar(lhs: PlironIntTensor, rhs: burn_std::Scalar) -> PlironIntTensor {
        todo!()
    }

    fn int_div(lhs: PlironIntTensor, rhs: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn int_div_scalar(lhs: PlironIntTensor, rhs: burn_std::Scalar) -> PlironIntTensor {
        todo!()
    }

    fn int_remainder(lhs: PlironIntTensor, rhs: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn int_remainder_scalar(lhs: PlironIntTensor, rhs: burn_std::Scalar) -> PlironIntTensor {
        todo!()
    }

    fn int_matmul(lhs: PlironIntTensor, rhs: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn int_sum(tensor: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn int_sum_dim(tensor: PlironIntTensor, dim: usize) -> PlironIntTensor {
        todo!()
    }

    fn int_prod(tensor: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn int_prod_dim(tensor: PlironIntTensor, dim: usize) -> PlironIntTensor {
        todo!()
    }

    fn int_mean_dim(tensor: PlironIntTensor, dim: usize) -> PlironIntTensor {
        todo!()
    }

    fn int_cumsum(tensor: PlironIntTensor, dim: usize) -> PlironIntTensor {
        todo!()
    }

    fn int_cumprod(tensor: PlironIntTensor, dim: usize) -> PlironIntTensor {
        todo!()
    }

    fn int_cummin(tensor: PlironIntTensor, dim: usize) -> PlironIntTensor {
        todo!()
    }

    fn int_cummax(tensor: PlironIntTensor, dim: usize) -> PlironIntTensor {
        todo!()
    }

    fn int_argmax(tensor: PlironIntTensor, dim: usize) -> PlironIntTensor {
        todo!()
    }

    fn int_argmin(tensor: PlironIntTensor, dim: usize) -> PlironIntTensor {
        todo!()
    }

    fn int_argtopk(tensor: PlironIntTensor, dim: usize, k: usize) -> PlironIntTensor {
        todo!()
    }

    fn int_abs(tensor: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn int_swap_dims(tensor: PlironIntTensor, dim1: usize, dim2: usize) -> PlironIntTensor {
        todo!()
    }

    fn int_permute(tensor: PlironIntTensor, axes: &[usize]) -> PlironIntTensor {
        todo!()
    }

    fn int_flip(tensor: PlironIntTensor, axes: &[usize]) -> PlironIntTensor {
        todo!()
    }

    fn int_random(
        shape: Shape,
        distribution: burn_std::Distribution,
        device: &PlironDevice,
        dtype: burn_std::IntDType,
    ) -> PlironIntTensor {
        todo!()
    }

    fn int_expand(tensor: PlironIntTensor, shape: Shape) -> PlironIntTensor {
        todo!()
    }

    fn bitwise_and(lhs: PlironIntTensor, rhs: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn bitwise_and_scalar(lhs: PlironIntTensor, rhs: burn_std::Scalar) -> PlironIntTensor {
        todo!()
    }

    fn bitwise_or(lhs: PlironIntTensor, rhs: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn bitwise_or_scalar(lhs: PlironIntTensor, rhs: burn_std::Scalar) -> PlironIntTensor {
        todo!()
    }

    fn bitwise_xor(lhs: PlironIntTensor, rhs: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn bitwise_xor_scalar(lhs: PlironIntTensor, rhs: burn_std::Scalar) -> PlironIntTensor {
        todo!()
    }

    fn bitwise_not(tensor: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn bitwise_left_shift(lhs: PlironIntTensor, rhs: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn bitwise_left_shift_scalar(lhs: PlironIntTensor, rhs: burn_std::Scalar) -> PlironIntTensor {
        todo!()
    }

    fn bitwise_right_shift(lhs: PlironIntTensor, rhs: PlironIntTensor) -> PlironIntTensor {
        todo!()
    }

    fn bitwise_right_shift_scalar(lhs: PlironIntTensor, rhs: burn_std::Scalar) -> PlironIntTensor {
        todo!()
    }

    fn int_cast(tensor: PlironIntTensor, dtype: burn_std::IntDType) -> PlironIntTensor {
        todo!()
    }

    fn int_unfold(
        tensor: PlironIntTensor,
        dim: usize,
        size: usize,
        step: usize,
    ) -> PlironIntTensor {
        todo!()
    }
}

impl<F: Send + Sync + 'static, I: Send + Sync + 'static, B: Send + Sync + 'static>
    ActivationOps<Pliron<F, I, B>> for Pliron<F, I, B>
{
}

impl<F: Send + Sync + 'static, I: Send + Sync + 'static, B: Send + Sync + 'static>
    TransactionOps<Pliron<F, I, B>> for Pliron<F, I, B>
{
}

impl<F: Send + Sync + 'static, I: Send + Sync + 'static, B: Send + Sync + 'static>
    QTensorOps<Pliron<F, I, B>> for Pliron<F, I, B>
{
    fn q_from_data(data: TensorData, device: &PlironDevice) -> PlironQTensor {
        todo!()
    }

    fn quantize(
        tensor: PlironFloatTensor,
        scheme: &burn_std::quantization::QuantScheme,
        qparams: burn_backend::quantization::QuantizationParametersPrimitive<Pliron<F, I, B>>,
    ) -> PlironQTensor {
        todo!()
    }

    fn dequantize(tensor: PlironQTensor, out_dtype: burn_std::FloatDType) -> PlironFloatTensor {
        todo!()
    }

    fn q_device(tensor: &PlironQTensor) -> PlironDevice {
        todo!()
    }

    fn q_to_device(tensor: PlironQTensor, device: &PlironDevice) -> PlironQTensor {
        todo!()
    }

    fn q_reshape(tensor: PlironQTensor, shape: Shape) -> PlironQTensor {
        todo!()
    }

    async fn q_into_data(
        tensor: PlironQTensor,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        todo!()
    }

    fn q_expand(tensor: PlironQTensor, shape: Shape) -> PlironQTensor {
        todo!()
    }

    fn q_swap_dims(tensor: PlironQTensor, dim1: usize, dim2: usize) -> PlironQTensor {
        todo!()
    }

    fn q_permute(tensor: PlironQTensor, axes: &[usize]) -> PlironQTensor {
        todo!()
    }

    fn q_flip(tensor: PlironQTensor, axes: &[usize]) -> PlironQTensor {
        todo!()
    }

    fn q_select(tensor: PlironQTensor, dim: usize, indices: PlironIntTensor) -> PlironQTensor {
        todo!()
    }

    fn q_slice(tensor: PlironQTensor, slices: &[burn_std::Slice]) -> PlironQTensor {
        todo!()
    }
}

impl<F: Send + Sync + 'static, I: Send + Sync + 'static, B: Send + Sync + 'static>
    ModuleOps<Pliron<F, I, B>> for Pliron<F, I, B>
{
    fn rfft(
        tensor: PlironFloatTensor,
        n: usize,
        dim: Option<usize>,
    ) -> (PlironFloatTensor, PlironFloatTensor) {
        todo!()
    }

    fn irfft(
        real: PlironFloatTensor,
        imag: PlironFloatTensor,
        n: usize,
        dim: Option<usize>,
    ) -> PlironFloatTensor {
        todo!()
    }
    fn conv2d(
        x: PlironFloatTensor,
        weight: PlironFloatTensor,
        bias: Option<PlironFloatTensor>,
        options: burn_std::ops::ConvOptions<2>,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn deform_conv2d(
        x: PlironFloatTensor,
        offset: PlironFloatTensor,
        weight: PlironFloatTensor,
        mask: Option<PlironFloatTensor>,
        bias: Option<PlironFloatTensor>,
        options: burn_std::ops::DeformConvOptions<2>,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn deform_conv2d_backward(
        x: PlironFloatTensor,
        offset: PlironFloatTensor,
        weight: PlironFloatTensor,
        mask: Option<PlironFloatTensor>,
        bias: Option<PlironFloatTensor>,
        output_grad: PlironFloatTensor,
        options: burn_std::ops::DeformConvOptions<2>,
    ) -> burn_backend::ops::DeformConv2dBackward<Pliron<F, I, B>> {
        todo!()
    }

    fn conv3d(
        x: PlironFloatTensor,
        weight: PlironFloatTensor,
        bias: Option<PlironFloatTensor>,
        options: burn_std::ops::ConvOptions<3>,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn conv_transpose2d(
        x: PlironFloatTensor,
        weight: PlironFloatTensor,
        bias: Option<PlironFloatTensor>,
        options: burn_std::ops::ConvTransposeOptions<2>,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn conv_transpose3d(
        x: PlironFloatTensor,
        weight: PlironFloatTensor,
        bias: Option<PlironFloatTensor>,
        options: burn_std::ops::ConvTransposeOptions<3>,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn avg_pool2d(
        x: PlironFloatTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn avg_pool2d_backward(
        x: PlironFloatTensor,
        grad: PlironFloatTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn adaptive_avg_pool2d(x: PlironFloatTensor, output_size: [usize; 2]) -> PlironFloatTensor {
        todo!()
    }

    fn adaptive_avg_pool2d_backward(
        x: PlironFloatTensor,
        grad: PlironFloatTensor,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn max_pool2d(
        x: PlironFloatTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn max_pool2d_with_indices(
        x: PlironFloatTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> burn_backend::ops::MaxPool2dWithIndices<Pliron<F, I, B>> {
        todo!()
    }

    fn max_pool2d_with_indices_backward(
        x: PlironFloatTensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: PlironFloatTensor,
        indices: PlironIntTensor,
    ) -> burn_backend::ops::MaxPool2dBackward<Pliron<F, I, B>> {
        todo!()
    }

    fn interpolate(
        x: PlironFloatTensor,
        output_size: [usize; 2],
        options: burn_std::ops::InterpolateOptions,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn interpolate_backward(
        x: PlironFloatTensor,
        grad: PlironFloatTensor,
        output_size: [usize; 2],
        options: burn_std::ops::InterpolateOptions,
    ) -> PlironFloatTensor {
        todo!()
    }

    fn attention(
        query: PlironFloatTensor,
        key: PlironFloatTensor,
        value: PlironFloatTensor,
        mask: Option<PlironBoolTensor>,
        attn_bias: Option<PlironFloatTensor>,
        options: burn_std::ops::AttentionModuleOptions,
    ) -> PlironFloatTensor {
        todo!()
    }
}
