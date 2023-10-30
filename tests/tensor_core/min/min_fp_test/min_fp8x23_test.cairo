// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion::test_helper::tensor::fixed_point::fp8x23::fp_tensor_1x3_helper;
    use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    use orion::numbers::fixed_point::core::{FixedTrait};

    #[test]
    #[available_gas(2000000)]
    fn tensor_min() {
        let tensor = fp_tensor_1x3_helper();

        let result = tensor.min().mag;
        assert(result == 0, 'tensor.min = 0');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion::test_helper::tensor::fixed_point::fp8x23::fp_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    use orion::numbers::fixed_point::core::{FixedTrait};


    #[test]
    #[available_gas(2000000)]
    fn tensor_min() {
        let tensor = fp_tensor_2x2_helper();

        let result = tensor.min().mag;
        assert(result == 0, 'tensor.min = 0');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion::test_helper::tensor::fixed_point::fp8x23::fp_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    use orion::numbers::fixed_point::core::{FixedTrait};


    #[test]
    #[available_gas(2000000)]
    fn tensor_min() {
        let tensor = fp_tensor_2x2x2_helper();

        let result = tensor.min().mag;
        assert(result == 0, 'tensor.min = 0');
    }
}

