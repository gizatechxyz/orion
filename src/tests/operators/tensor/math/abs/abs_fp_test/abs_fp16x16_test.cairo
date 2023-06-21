// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_neg_helper;
    use orion::numbers::fixed_point::core::{FixedTrait};
    use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

    #[test]
    #[available_gas(2000000000)]
    fn tensor_abs() {
        let tensor = fp_tensor_1x3_neg_helper();
        let result = tensor.abs();
        assert(*result.data[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result.data[1] == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert(*result.data[2] == FixedTrait::new_unscaled(2, false), 'result[0] = 2');
    }
}


// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_neg_helper;
    use orion::numbers::fixed_point::core::{FixedTrait};
    use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

    #[test]
    #[available_gas(2000000000)]
    fn tensor_abs() {
        let tensor = fp_tensor_2x2_neg_helper();
        let result = tensor.abs();
        assert(*result.data[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result.data[1] == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert(*result.data[2] == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert(*result.data[3] == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_neg_helper;
    use orion::numbers::fixed_point::core::{FixedTrait};
    use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

    #[test]
    #[available_gas(2000000000)]
    fn tensor_abs() {
        let tensor = fp_tensor_2x2x2_neg_helper();
        let result = tensor.abs();

        assert(*result.data[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result.data[1] == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert(*result.data[2] == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert(*result.data[3] == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
        assert(*result.data[4] == FixedTrait::new_unscaled(4, false), 'result[4] = 4');
        assert(*result.data[5] == FixedTrait::new_unscaled(5, false), 'result[5] = 5');
        assert(*result.data[6] == FixedTrait::new_unscaled(6, false), 'result[6] = 6');
        assert(*result.data[7] == FixedTrait::new_unscaled(7, false), 'result[7] = 7');
    }
}
