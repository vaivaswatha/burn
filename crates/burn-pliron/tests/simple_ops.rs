//! Simple ops tests for pliron backend

#[cfg(test)]
mod tests {
    use burn_pliron::PlironDevice;
    type PlironTensor<const D: usize> = burn_tensor::Tensor<burn_pliron::PlironBackend, D>;

    #[test]
    fn test_simple() {
        let device = PlironDevice::default();
        let a = PlironTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let a_ret = a.to_data();
        assert!(a_ret.shape == [2, 2]);
        assert_eq!(a_ret.to_string(), "[1.0, 2.0, 3.0, 4.0]");
    }

    #[test]
    fn test_add() {
        let device = PlironDevice::default();
        let a = PlironTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let b = PlironTensor::<2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);
        let c = a + b;
        let c_data = c.to_data();
        assert!(c_data.shape == [2, 2]);
        assert_eq!(c_data.to_string(), "[6.0, 8.0, 10.0, 12.0]");
    }
}
