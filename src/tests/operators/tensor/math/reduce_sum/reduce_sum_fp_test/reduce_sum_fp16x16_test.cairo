// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::TensorTrait;
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::{
        FP16x16Into, FP16x16Impl, FP16x16PartialEq
    };
    use orion::numbers::fixed_point::core::FixedTrait;

    #[test]
    #[available_gas(20000000)]
    fn reduce_sum() {
        let tensor = fp_tensor_1x3_helper();

        let result = tensor.reduce_sum(0, false);
        assert((*result.data[0]) == FixedTrait::new_unscaled(3, false), 'result[0] = 3');
        assert(result.data.len() == 1, 'result.data.len = 1');
    }

    #[test]
    #[should_panic(expected: ('axis out of dimensions', ))]
    #[available_gas(20000000)]
    fn out_of_dim() {
        let tensor = fp_tensor_1x3_helper();

        let result = tensor.reduce_sum(1, false);
    }
}
// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::TensorTrait;
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::{
        FP16x16Into, FP16x16Impl, FP16x16PartialEq
    };
    use orion::numbers::fixed_point::core::FixedTrait;


    #[test]
    #[available_gas(20000000)]
    fn axis_0_keepdims_false() {
        let tensor = fp_tensor_2x2_helper();

        let result = tensor.reduce_sum(0, false);
        assert((*result.data[0]) == FixedTrait::new_unscaled(2, false), 'result.data[0] = 2');
        assert((*result.data[1]) == FixedTrait::new_unscaled(4, false), 'result.data[1] = 4');
        assert((*result.shape[0]) == 2, 'result.shape[0] = 2');
        assert(result.data.len() == 2, 'result.data.len = 1');
        assert(result.shape.len() == 1, 'result.shape.len = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_0_keepdims_true() {
        let tensor = fp_tensor_2x2_helper();

        let result = tensor.reduce_sum(0, true);

        assert((*result.data[0]) == FixedTrait::new_unscaled(2, false), 'result.data[0] = 2');
        assert((*result.data[1]) == FixedTrait::new_unscaled(4, false), 'result.data[1] = 4');
        assert((*result.shape[0]) == 1, 'result.shape[0] = 1');
        assert((*result.shape[1]) == 2, 'result.shape[1] = 2');
        assert(result.data.len() == 2, 'result.data.len = 1');
        assert(result.shape.len() == 2, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_1_keepdims_false() {
        let tensor = fp_tensor_2x2_helper();

        let result = tensor.reduce_sum(1, false);

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result.data[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(5, false), 'result.data[1] = 5');
        assert((*result.shape[0]) == 2, 'result.shape[0] = 2');
        assert(result.data.len() == 2, 'result.data.len = 1');
        assert(result.shape.len() == 1, 'result.shape.len = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_1_keepdims_true() {
        let tensor = fp_tensor_2x2_helper();

        let result = tensor.reduce_sum(1, true);

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result.data[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(5, false), 'result.data[1] = 5');
        assert((*result.shape[0]) == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]) == 1, 'result.shape[1] = 1');
        assert(result.data.len() == 2, 'result.data.len = 1');
        assert(result.shape.len() == 2, 'result.shape.len = 2');
    }

    #[test]
    #[should_panic(expected: ('axis out of dimensions', ))]
    #[available_gas(20000000)]
    fn out_of_dim() {
        let tensor = fp_tensor_2x2_helper();

        let result = tensor.reduce_sum(2, false);
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::TensorTrait;
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::{
        FP16x16Into, FP16x16Impl, FP16x16PartialEq
    };
    use orion::numbers::fixed_point::core::FixedTrait;


    #[test]
    #[available_gas(20000000)]
    fn axis_0_keepdims_false() {
        let tensor = fp_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(0, false);
        assert((*result.data[0]) == FixedTrait::new_unscaled(4, false), 'result[0] = 4');
        assert((*result.data[1]) == FixedTrait::new_unscaled(6, false), 'result[1] = 6');
        assert((*result.data[2]) == FixedTrait::new_unscaled(8, false), 'result[2] = 8');
        assert((*result.data[3]) == FixedTrait::new_unscaled(10, false), 'result[3] = 10');
        assert((*result.shape[0]) == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]) == 2, 'result.shape[1] = 2');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 2, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_0_keepdims_true() {
        let tensor = fp_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(0, true);
        assert((*result.data[0]) == FixedTrait::new_unscaled(4, false), 'result[0] = 4');
        assert((*result.data[1]) == FixedTrait::new_unscaled(6, false), 'result[1] = 6');
        assert((*result.data[2]) == FixedTrait::new_unscaled(8, false), 'result[2] = 8');
        assert((*result.data[3]) == FixedTrait::new_unscaled(10, false), 'result[3] = 10');
        assert((*result.shape[0]) == 1, 'result.shape[0] = 1');
        assert((*result.shape[1]) == 2, 'result.shape[1] = 2');
        assert((*result.shape[2]) == 2, 'result.shape[2] = 2');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 3, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_1_keepdims_false() {
        let tensor = fp_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(1, false);
        assert((*result.data[0]) == FixedTrait::new_unscaled(2, false), 'result[0] = 2');
        assert((*result.data[1]) == FixedTrait::new_unscaled(4, false), 'result[1] = 4');
        assert((*result.data[2]) == FixedTrait::new_unscaled(10, false), 'result[2] = 10');
        assert((*result.data[3]) == FixedTrait::new_unscaled(12, false), 'result[3] = 12');
        assert((*result.shape[0]) == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]) == 2, 'result.shape[1] = 2');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 2, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_1_keepdims_true() {
        let tensor = fp_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(1, true);
        assert((*result.data[0]) == FixedTrait::new_unscaled(2, false), 'result[0] = 2');
        assert((*result.data[1]) == FixedTrait::new_unscaled(4, false), 'result[1] = 4');
        assert((*result.data[2]) == FixedTrait::new_unscaled(10, false), 'result[2] = 10');
        assert((*result.data[3]) == FixedTrait::new_unscaled(12, false), 'result[3] = 12');
        assert((*result.shape[0]) == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]) == 1, 'result.shape[1] = 1');
        assert((*result.shape[2]) == 2, 'result.shape[2] = 2');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 3, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_2_keepdims_false() {
        let tensor = fp_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(2, false);
        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(5, false), 'result[1] = 5');
        assert((*result.data[2]) == FixedTrait::new_unscaled(9, false), 'result[2] = 9');
        assert((*result.data[3]) == FixedTrait::new_unscaled(13, false), 'result[3] = 13');
        assert((*result.shape[0]) == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]) == 2, 'result.shape[1] = 2');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 2, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_2_keepdims_true() {
        let tensor = fp_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(2, true);
        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(5, false), 'result[1] = 5');
        assert((*result.data[2]) == FixedTrait::new_unscaled(9, false), 'result[2] = 9');
        assert((*result.data[3]) == FixedTrait::new_unscaled(13, false), 'result[3] = 13');
        assert((*result.shape[0]) == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]) == 2, 'result.shape[1] = 2');
        assert((*result.shape[2]) == 1, 'result.shape[2] = 1');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 3, 'result.shape.len = 2');
    }

    #[test]
    #[should_panic(expected: ('axis out of dimensions', ))]
    #[available_gas(20000000)]
    fn out_of_dim() {
        let tensor = fp_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(3, false);
    }
}

