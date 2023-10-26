// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use orion::operators::tensor::I32Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion_tests::helpers::tensor::i32::i32_tensor_1x3_helper;

    #[test]
    #[available_gas(2000000)]
    fn tensor_min() {
        let tensor = i32_tensor_1x3_helper();

        let result = tensor.min_in_tensor().mag;
        assert(result == 0, 'tensor.min = 0');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use orion::operators::tensor::I32Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion_tests::helpers::tensor::i32::i32_tensor_2x2_helper;

    #[test]
    #[available_gas(2000000)]
    fn tensor_min() {
        let tensor = i32_tensor_2x2_helper();

        let result = tensor.min_in_tensor().mag;
        assert(result == 0, 'tensor.min = 0');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use orion::operators::tensor::I32Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion_tests::helpers::tensor::i32::i32_tensor_2x2x2_helper;

    #[test]
    #[available_gas(2000000)]
    fn tensor_min() {
        let tensor = i32_tensor_2x2x2_helper();

        let result = tensor.min_in_tensor().mag;
        assert(result == 0, 'tensor.min = 0');
    }
}

