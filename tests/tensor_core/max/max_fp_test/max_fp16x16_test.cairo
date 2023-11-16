// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion::test_helper::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;
    use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
    use orion::numbers::fixed_point::core::{FixedTrait};

    #[test]
    #[available_gas(2000000)]
    fn tensor_max() {
        let tensor = fp_tensor_1x3_helper();

        let result = tensor.max_in_tensor().mag;
        assert(result == FixedTrait::<FP16x16>::new_unscaled(2, false).mag, 'tensor.max = 2');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion::test_helper::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
    use orion::numbers::fixed_point::core::{FixedTrait};


    #[test]
    #[available_gas(2000000)]
    fn tensor_max() {
        let tensor = fp_tensor_2x2_helper();

        let result = tensor.max_in_tensor().mag;
        assert(result == FixedTrait::<FP16x16>::new_unscaled(3, false).mag, 'tensor.max = 3');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::TensorTrait;
    use orion::test_helper::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
    use orion::numbers::fixed_point::core::{FixedTrait};


    #[test]
    #[available_gas(2000000)]
    fn tensor_max() {
        let tensor = fp_tensor_2x2x2_helper();

        let result = tensor.max_in_tensor().mag;
        assert(result == FixedTrait::<FP16x16>::new_unscaled(7, false).mag, 'tensor.max = 7');
    }
}

