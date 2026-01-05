//! Simple ops tests for pliron backend, mostly comparing with the ndarray backend.

#[cfg(test)]
mod tests {
    type GoldBackend = burn_ndarray::NdArray<f32>;
    type GoldTensor<const D: usize> = burn_tensor::Tensor<GoldBackend, D>;
    use burn_ndarray::NdArrayDevice as GoldDevice;

    #[test]
    fn test_add() {
        let device = GoldDevice::Cpu;
        let a = GoldTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let b = GoldTensor::<2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);
        let c = a + b;
        let expected = GoldTensor::<2>::from_data([[6.0, 8.0], [10.0, 12.0]], &device);
        assert_eq!(c.to_data(), expected.to_data());
    }
}
