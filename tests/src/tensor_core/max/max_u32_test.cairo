// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use orion::operators::tensor::U32Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion_tests::helpers::tensor::u32::u32_tensor_1x3_helper;

    #[test]
    #[available_gas(2000000)]
    fn tensor_max() {
        let tensor = u32_tensor_1x3_helper();

        let result = tensor.max();
        assert(result == 2, 'tensor.max = 2');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use orion::operators::tensor::U32Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion_tests::helpers::tensor::u32::u32_tensor_2x2_helper;

    #[test]
    #[available_gas(2000000)]
    fn tensor_max() {
        let tensor = u32_tensor_2x2_helper();

        let result = tensor.max();
        assert(result == 3, 'tensor.max = 3');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use orion::operators::tensor::U32Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion_tests::helpers::tensor::u32::u32_tensor_2x2x2_helper;

    #[test]
    #[available_gas(2000000)]
    fn tensor_max() {
        let tensor = u32_tensor_2x2x2_helper();

        let result = tensor.max();
        assert(result == 7, 'tensor.max = 7');
    }
}

