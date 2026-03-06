//! pliron backend for burn

use burn_tensor::{
    TensorMetadata,
    backend::{Backend, DTypeUsage, Device, DeviceOps, QTensorPrimitive},
    quantization::QuantScheme,
};

pub mod ir_interface;
pub mod tensor_ops;

#[derive(Debug, Default, Clone)]
pub struct PlironBackend;

impl Backend for PlironBackend {
    type Device = PlironDevice;

    type FloatTensorPrimitive = PlironFloatTensor;

    type FloatElem = f64;

    type IntTensorPrimitive = PlironIntTensor;

    type IntElem = u64;

    type BoolTensorPrimitive = PlironBoolTensor;

    type BoolElem = bool;

    type QuantizedTensorPrimitive = PlironQTensor;

    fn name(_device: &Self::Device) -> String {
        "Pliron".to_string()
    }

    fn seed(_device: &Self::Device, _seed: u64) {}

    fn dtype_usage(
        _device: &Self::Device,
        _dtype: burn_tensor::DType,
    ) -> burn_tensor::backend::DTypeUsageSet {
        DTypeUsage::general()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PlironDevice;

impl Device for PlironDevice {
    fn from_id(_device_id: burn_tensor::backend::DeviceId) -> Self {
        Self
    }

    fn to_id(&self) -> burn_tensor::backend::DeviceId {
        burn_tensor::backend::DeviceId {
            type_id: 0,
            index_id: 0,
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}

impl DeviceOps for PlironDevice {}

#[derive(Clone, Debug)]
pub struct PlironFloatTensor {
    pub shape: burn_tensor::Shape,
    pub value: pliron::value::Value,
}

impl TensorMetadata for PlironFloatTensor {
    fn dtype(&self) -> burn_tensor::DType {
        burn_tensor::DType::F64
    }

    fn shape(&self) -> burn_tensor::Shape {
        self.shape.clone()
    }
}

#[derive(Debug, Clone)]
pub struct PlironIntTensor {
    pub shape: burn_tensor::Shape,
    pub value: pliron::value::Value,
}

impl TensorMetadata for PlironIntTensor {
    fn dtype(&self) -> burn_tensor::DType {
        burn_tensor::DType::U64
    }

    fn shape(&self) -> burn_tensor::Shape {
        self.shape.clone()
    }
}

#[derive(Debug, Clone)]
pub struct PlironQTensor {
    pub shape: burn_tensor::Shape,
    pub scheme: QuantScheme,
    pub value: pliron::value::Value,
}

impl TensorMetadata for PlironQTensor {
    fn dtype(&self) -> burn_tensor::DType {
        burn_tensor::DType::QFloat(self.scheme)
    }

    fn shape(&self) -> burn_tensor::Shape {
        self.shape.clone()
    }
}

impl QTensorPrimitive for PlironQTensor {
    fn scheme(&self) -> &burn_tensor::quantization::QuantScheme {
        &self.scheme
    }
}

#[derive(Debug, Clone)]
pub struct PlironBoolTensor {
    pub shape: burn_tensor::Shape,
    pub value: pliron::value::Value,
}

impl TensorMetadata for PlironBoolTensor {
    fn dtype(&self) -> burn_tensor::DType {
        burn_tensor::DType::Bool
    }

    fn shape(&self) -> burn_tensor::Shape {
        self.shape.clone()
    }
}
