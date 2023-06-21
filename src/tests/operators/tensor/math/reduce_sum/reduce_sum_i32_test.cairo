// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::TensorTrait;
    use orion::tests::helpers::tensor::i32::i32_tensor_1x3_helper;

    #[test]
    #[available_gas(20000000)]
    fn reduce_sum() {
        let tensor = i32_tensor_1x3_helper();

        let result = tensor.reduce_sum(0, false);
        assert((*result.data[0]).into() == 3, 'result[0] = 3');
        assert(result.data.len() == 1, 'result.data.len = 1');
    }

    #[test]
    #[should_panic(expected: ('axis out of dimensions', ))]
    #[available_gas(20000000)]
    fn out_of_dim() {
        let tensor = i32_tensor_1x3_helper();

        let result = tensor.reduce_sum(1, false);
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::TensorTrait;
    use orion::tests::helpers::tensor::i32::i32_tensor_2x2_helper;

    #[test]
    #[available_gas(20000000)]
    fn axis_0_keepdims_false() {
        let tensor = i32_tensor_2x2_helper();

        let result = tensor.reduce_sum(0, false);
        assert((*result.data[0]).into() == 2, 'result.data[0] = 2');
        assert((*result.data[1]).into() == 4, 'result.data[1] = 4');
        assert((*result.shape[0]).into() == 2, 'result.shape[0] = 2');
        assert(result.data.len() == 2, 'result.data.len = 1');
        assert(result.shape.len() == 1, 'result.shape.len = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_0_keepdims_true() {
        let tensor = i32_tensor_2x2_helper();

        let result = tensor.reduce_sum(0, true);

        assert((*result.data[0]).into() == 2, 'result.data[0] = 2');
        assert((*result.data[1]).into() == 4, 'result.data[1] = 4');
        assert((*result.shape[0]).into() == 1, 'result.shape[0] = 1');
        assert((*result.shape[1]).into() == 2, 'result.shape[1] = 2');
        assert(result.data.len() == 2, 'result.data.len = 1');
        assert(result.shape.len() == 2, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_1_keepdims_false() {
        let tensor = i32_tensor_2x2_helper();

        let result = tensor.reduce_sum(1, false);

        assert((*result.data[0]).into() == 1, 'result.data[0] = 1');
        assert((*result.data[1]).into() == 5, 'result.data[1] = 5');
        assert((*result.shape[0]).into() == 2, 'result.shape[0] = 2');
        assert(result.data.len() == 2, 'result.data.len = 1');
        assert(result.shape.len() == 1, 'result.shape.len = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_1_keepdims_true() {
        let tensor = i32_tensor_2x2_helper();

        let result = tensor.reduce_sum(1, true);

        assert((*result.data[0]).into() == 1, 'result.data[0] = 1');
        assert((*result.data[1]).into() == 5, 'result.data[1] = 5');
        assert((*result.shape[0]).into() == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]).into() == 1, 'result.shape[1] = 1');
        assert(result.data.len() == 2, 'result.data.len = 1');
        assert(result.shape.len() == 2, 'result.shape.len = 2');
    }

    #[test]
    #[should_panic(expected: ('axis out of dimensions', ))]
    #[available_gas(20000000)]
    fn out_of_dim() {
        let tensor = i32_tensor_2x2_helper();

        let result = tensor.reduce_sum(2, false);
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::TensorTrait;
    use orion::tests::helpers::tensor::i32::i32_tensor_2x2x2_helper;

    #[test]
    #[available_gas(20000000)]
    fn axis_0_keepdims_false() {
        let tensor = i32_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(0, false);
        assert((*result.data[0]).into() == 4, 'result[0] = 4');
        assert((*result.data[1]).into() == 6, 'result[1] = 6');
        assert((*result.data[2]).into() == 8, 'result[2] = 8');
        assert((*result.data[3]).into() == 10, 'result[3] = 10');
        assert((*result.shape[0]).into() == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]).into() == 2, 'result.shape[1] = 2');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 2, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_0_keepdims_true() {
        let tensor = i32_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(0, true);
        assert((*result.data[0]).into() == 4, 'result[0] = 4');
        assert((*result.data[1]).into() == 6, 'result[1] = 6');
        assert((*result.data[2]).into() == 8, 'result[2] = 8');
        assert((*result.data[3]).into() == 10, 'result[3] = 10');
        assert((*result.shape[0]).into() == 1, 'result.shape[0] = 1');
        assert((*result.shape[1]).into() == 2, 'result.shape[1] = 2');
        assert((*result.shape[2]).into() == 2, 'result.shape[2] = 2');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 3, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_1_keepdims_false() {
        let tensor = i32_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(1, false);
        assert((*result.data[0]).into() == 2, 'result[0] = 2');
        assert((*result.data[1]).into() == 4, 'result[1] = 4');
        assert((*result.data[2]).into() == 10, 'result[2] = 10');
        assert((*result.data[3]).into() == 12, 'result[3] = 12');
        assert((*result.shape[0]).into() == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]).into() == 2, 'result.shape[1] = 2');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 2, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_1_keepdims_true() {
        let tensor = i32_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(1, true);
        assert((*result.data[0]).into() == 2, 'result[0] = 2');
        assert((*result.data[1]).into() == 4, 'result[1] = 4');
        assert((*result.data[2]).into() == 10, 'result[2] = 10');
        assert((*result.data[3]).into() == 12, 'result[3] = 12');
        assert((*result.shape[0]).into() == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]).into() == 1, 'result.shape[1] = 1');
        assert((*result.shape[2]).into() == 2, 'result.shape[2] = 2');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 3, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_2_keepdims_false() {
        let tensor = i32_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(2, false);
        assert((*result.data[0]).into() == 1, 'result[0] = 1');
        assert((*result.data[1]).into() == 5, 'result[1] = 5');
        assert((*result.data[2]).into() == 9, 'result[2] = 9');
        assert((*result.data[3]).into() == 13, 'result[3] = 13');
        assert((*result.shape[0]).into() == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]).into() == 2, 'result.shape[1] = 2');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 2, 'result.shape.len = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn axis_2_keepdims_true() {
        let tensor = i32_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(2, true);
        assert((*result.data[0]).into() == 1, 'result[0] = 1');
        assert((*result.data[1]).into() == 5, 'result[1] = 5');
        assert((*result.data[2]).into() == 9, 'result[2] = 9');
        assert((*result.data[3]).into() == 13, 'result[3] = 13');
        assert((*result.shape[0]).into() == 2, 'result.shape[0] = 2');
        assert((*result.shape[1]).into() == 2, 'result.shape[1] = 2');
        assert((*result.shape[2]).into() == 1, 'result.shape[2] = 1');
        assert(result.data.len() == 4, 'result.data.len = 4');
        assert(result.shape.len() == 3, 'result.shape.len = 2');
    }

    #[test]
    #[should_panic(expected: ('axis out of dimensions', ))]
    #[available_gas(20000000)]
    fn out_of_dim() {
        let tensor = i32_tensor_2x2x2_helper();

        let result = tensor.reduce_sum(3, false);
    }
}
