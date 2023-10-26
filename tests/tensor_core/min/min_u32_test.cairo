// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use orion::operators::tensor::U32Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion::helpers::tensor::u32::u32_tensor_1x3_helper;

    #[test]
    #[available_gas(2000000)]
    fn tensor_min() {
        let tensor = u32_tensor_1x3_helper();

        let result = tensor.min();
        assert(result == 0, 'tensor.min = 0');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use orion::operators::tensor::U32Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion::helpers::tensor::u32::u32_tensor_2x2_helper;

    #[test]
    #[available_gas(2000000)]
    fn tensor_min() {
        let tensor = u32_tensor_2x2_helper();

        let result = tensor.min();
        assert(result == 0, 'tensor.min = 0');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use orion::operators::tensor::U32Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion::helpers::tensor::u32::u32_tensor_2x2x2_helper;

    #[test]
    #[available_gas(2000000)]
    fn tensor_min() {
        let tensor = u32_tensor_2x2x2_helper();

        let result = tensor.min();
        assert(result == 0, 'tensor.min = 0');
    }
}

