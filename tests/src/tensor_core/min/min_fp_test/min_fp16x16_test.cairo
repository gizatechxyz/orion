// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion_tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;
    use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;
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
    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion_tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;
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
    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion_tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;
    use orion::numbers::fixed_point::core::{FixedTrait};


    #[test]
    #[available_gas(2000000)]
    fn tensor_min() {
        let tensor = fp_tensor_2x2x2_helper();

        let result = tensor.min().mag;
        assert(result == 0, 'tensor.min = 0');
    }
}

