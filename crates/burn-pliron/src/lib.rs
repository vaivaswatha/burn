//! pliron backend for burn

use core::marker::PhantomData;

use burn_backend::{
    Backend, BackendTypes, DTypeUsage, DeviceOps, QTensorPrimitive, TensorMetadata,
};
use burn_backend::quantization::QuantScheme;

pub mod ir_interface;
pub mod tensor_ops;

pub struct Pliron<F = f32, I = i32, B = u8>(PhantomData<(F, I, B)>);

pub type PlironBackend = Pliron<f32, i32, u8>;

impl<F, I, B> Clone for Pliron<F, I, B> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<F, I, B> Default for Pliron<F, I, B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<F, I, B> core::fmt::Debug for Pliron<F, I, B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Pliron")
    }
}

impl<F: Send + Sync + 'static, I: Send + Sync + 'static, B: Send + Sync + 'static> BackendTypes for Pliron<F, I, B> {
    type Device = PlironDevice;

    type FloatTensorPrimitive = PlironFloatTensor;

    type FloatElem = f64;

    type IntTensorPrimitive = PlironIntTensor;

    type IntElem = u64;

    type BoolTensorPrimitive = PlironBoolTensor;

    type BoolElem = bool;

    type QuantizedTensorPrimitive = PlironQTensor;
}

impl<F: Send + Sync + 'static, I: Send + Sync + 'static, B: Send + Sync + 'static> Backend for Pliron<F, I, B> {
    fn name(_device: &Self::Device) -> String {
        "Pliron".to_string()
    }

    fn seed(_device: &Self::Device, _seed: u64) {}

    fn dtype_usage(
        _device: &Self::Device,
        _dtype: burn_backend::DType,
    ) -> burn_backend::DTypeUsageSet {
        DTypeUsage::general()
    }

    fn device_count(_: u16) -> usize {
        1
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PlironDevice;

impl burn_backend::Device for PlironDevice {
    fn from_id(_device_id: burn_backend::DeviceId) -> Self {
        Self
    }

    fn to_id(&self) -> burn_backend::DeviceId {
        burn_backend::DeviceId {
            type_id: 0,
            index_id: 0,
        }
    }
}

impl DeviceOps for PlironDevice {}

#[derive(Clone, Debug)]
pub struct PlironFloatTensor {
    pub shape: burn_std::Shape,
    pub value: pliron::value::Value,
}

impl TensorMetadata for PlironFloatTensor {
    fn dtype(&self) -> burn_std::DType {
        burn_std::DType::F32
    }

    fn shape(&self) -> burn_std::Shape {
        self.shape.clone()
    }
}

#[derive(Debug, Clone)]
pub struct PlironIntTensor {
    pub shape: burn_std::Shape,
    pub value: pliron::value::Value,
}

impl TensorMetadata for PlironIntTensor {
    fn dtype(&self) -> burn_std::DType {
        burn_std::DType::U64
    }

    fn shape(&self) -> burn_std::Shape {
        self.shape.clone()
    }
}

#[derive(Debug, Clone)]
pub struct PlironQTensor {
    pub shape: burn_std::Shape,
    pub scheme: QuantScheme,
    pub value: pliron::value::Value,
}

impl TensorMetadata for PlironQTensor {
    fn dtype(&self) -> burn_std::DType {
        burn_std::DType::QFloat(self.scheme)
    }

    fn shape(&self) -> burn_std::Shape {
        self.shape.clone()
    }
}

impl QTensorPrimitive for PlironQTensor {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }
}

#[derive(Debug, Clone)]
pub struct PlironBoolTensor {
    pub shape: burn_std::Shape,
    pub value: pliron::value::Value,
}

impl TensorMetadata for PlironBoolTensor {
    fn dtype(&self) -> burn_std::DType {
        burn_std::DType::Bool(burn_std::BoolStore::Native)
    }

    fn shape(&self) -> burn_std::Shape {
        self.shape.clone()
    }
}
