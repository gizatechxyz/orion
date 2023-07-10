// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;

    #[test]
    #[should_panic(expected: ('cannot transpose a 1D tensor', ))]
    #[available_gas(2000000)]
    fn tensor_transpose() {
        let mut axes: Array<usize> = ArrayTrait::new();
        axes.append(1);

        let tensor = fp_tensor_1x3_helper();

        let result = tensor.transpose(axes.span());
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp16x16::{
        fp_tensor_2x2_helper, fp_tensor_3x2_helper, fp_tensor_2x3_helper
    };
    use orion::numbers::fixed_point::core::FixedTrait;
    use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};


    #[test]
    #[available_gas(20000000)]
    fn tensor_transpose() {
        let mut axes: Array<usize> = ArrayTrait::new();
        axes.append(1);
        axes.append(0);

        let tensor = fp_tensor_2x2_helper();

        let result = tensor.transpose(axes.span());

        assert(*result.data[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result.data[1] == FixedTrait::new_unscaled(2, false), 'result[1] = 2');
        assert(*result.data[2] == FixedTrait::new_unscaled(1, false), 'result[2] = 1');
        assert(*result.data[3] == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
        assert(*result.shape[0] == 2, 'shape[0] = 2');
        assert(*result.shape[1] == 2, 'shape[1] = 2');

        let tensor = fp_tensor_3x2_helper();

        let result = tensor.transpose(axes.span());

        assert(*result.data[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result.data[1] == FixedTrait::new_unscaled(2, false), 'result[1] = 2');
        assert(*result.data[2] == FixedTrait::new_unscaled(4, false), 'result[2] = 4');
        assert(*result.data[3] == FixedTrait::new_unscaled(1, false), 'result[3] = 1');
        assert(*result.data[4] == FixedTrait::new_unscaled(3, false), 'result[4] = 3');
        assert(*result.data[5] == FixedTrait::new_unscaled(5, false), 'result[5] = 5');
        assert(*result.shape[0] == 2, 'shape[0] = 2');
        assert(*result.shape[1] == 3, 'shape[1] = 3');

        let tensor = fp_tensor_2x3_helper();

        let result = tensor.transpose(axes.span());

        assert(*result.data[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result.data[1] == FixedTrait::new_unscaled(3, false), 'result[1] = 3');
        assert(*result.data[2] == FixedTrait::new_unscaled(1, false), 'result[2] = 1');
        assert(*result.data[3] == FixedTrait::new_unscaled(4, false), 'result[3] = 4');
        assert(*result.data[4] == FixedTrait::new_unscaled(2, false), 'result[4] = 2');
        assert(*result.data[5] == FixedTrait::new_unscaled(5, false), 'result[5] = 5');
        assert(*result.shape[0] == 3, 'shape[0] = 3');
        assert(*result.shape[1] == 2, 'shape[1] = 2');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp16x16::{
        fp_tensor_2x2x2_helper, fp_tensor_3x2x2_helper, fp_tensor_2x3_helper
    };
    use orion::numbers::fixed_point::core::FixedTrait;
    use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};


    #[test]
    #[available_gas(20000000)]
    fn tensor_transpose() {
        let tensor = fp_tensor_2x2x2_helper();

        let mut axes: Array<usize> = ArrayTrait::new();
        axes.append(1);
        axes.append(2);
        axes.append(0);

        let result = tensor.transpose(axes.span()).data;

        assert(*result[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result[1] == FixedTrait::new_unscaled(4, false), 'result[1] = 4');
        assert(*result[2] == FixedTrait::new_unscaled(1, false), 'result[2] = 1');
        assert(*result[3] == FixedTrait::new_unscaled(5, false), 'result[3] = 5');
        assert(*result[4] == FixedTrait::new_unscaled(2, false), 'result[4] = 2');
        assert(*result[5] == FixedTrait::new_unscaled(6, false), 'result[5] = 6');
        assert(*result[6] == FixedTrait::new_unscaled(3, false), 'result[6] = 3');
        assert(*result[7] == FixedTrait::new_unscaled(7, false), 'result[7] = 7');

        let mut axes: Array<usize> = ArrayTrait::new();
        axes.append(2);
        axes.append(1);
        axes.append(0);

        let result = tensor.transpose(axes.span()).data;

        assert(*result[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result[1] == FixedTrait::new_unscaled(4, false), 'result[1] = 4');
        assert(*result[2] == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert(*result[3] == FixedTrait::new_unscaled(6, false), 'result[3] = 6');
        assert(*result[4] == FixedTrait::new_unscaled(1, false), 'result[4] = 1');
        assert(*result[5] == FixedTrait::new_unscaled(5, false), 'result[5] = 5');
        assert(*result[6] == FixedTrait::new_unscaled(3, false), 'result[6] = 3');
        assert(*result[7] == FixedTrait::new_unscaled(7, false), 'result[7] = 7');

        let mut axes: Array<usize> = ArrayTrait::new();
        axes.append(0);
        axes.append(2);
        axes.append(1);

        let result = tensor.transpose(axes.span()).data;

        assert(*result[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result[1] == FixedTrait::new_unscaled(2, false), 'result[1] = 2');
        assert(*result[2] == FixedTrait::new_unscaled(1, false), 'result[2] = 1');
        assert(*result[3] == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
        assert(*result[4] == FixedTrait::new_unscaled(4, false), 'result[4] = 4');
        assert(*result[5] == FixedTrait::new_unscaled(6, false), 'result[5] = 6');
        assert(*result[6] == FixedTrait::new_unscaled(5, false), 'result[6] = 5');
        assert(*result[7] == FixedTrait::new_unscaled(7, false), 'result[7] = 7');

        let tensor = fp_tensor_3x2x2_helper();

        let mut axes: Array<usize> = ArrayTrait::new();
        axes.append(1);
        axes.append(2);
        axes.append(0);

        let result = tensor.transpose(axes.span());

        assert(*result.data[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result.data[1] == FixedTrait::new_unscaled(4, false), 'result[1] = 4');
        assert(*result.data[2] == FixedTrait::new_unscaled(8, false), 'result[2] = 8');
        assert(*result.data[3] == FixedTrait::new_unscaled(1, false), 'result[3] = 1');
        assert(*result.data[4] == FixedTrait::new_unscaled(5, false), 'result[4] = 5');
        assert(*result.data[5] == FixedTrait::new_unscaled(9, false), 'result[5] = 9');
        assert(*result.data[6] == FixedTrait::new_unscaled(2, false), 'result[6] = 2');
        assert(*result.data[7] == FixedTrait::new_unscaled(6, false), 'result[7] = 6');
        assert(*result.data[8] == FixedTrait::new_unscaled(10, false), 'result[8] = 10');
        assert(*result.data[9] == FixedTrait::new_unscaled(3, false), 'result[9] = 3');
        assert(*result.data[10] == FixedTrait::new_unscaled(7, false), 'result[10] = 7');
        assert(*result.data[11] == FixedTrait::new_unscaled(11, false), 'result[11] = 11');
        assert(*result.shape[0] == 2, 'shape[0] = 2');
        assert(*result.shape[1] == 2, 'shape[1] = 2');
        assert(*result.shape[2] == 3, 'shape[2] = 3');

        let mut axes: Array<usize> = ArrayTrait::new();
        axes.append(2);
        axes.append(1);
        axes.append(0);

        let result = tensor.transpose(axes.span());

        assert(*result.data[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result.data[1] == FixedTrait::new_unscaled(4, false), 'result[1] = 4');
        assert(*result.data[2] == FixedTrait::new_unscaled(8, false), 'result[2] = 8');
        assert(*result.data[3] == FixedTrait::new_unscaled(2, false), 'result[3] = 2');
        assert(*result.data[4] == FixedTrait::new_unscaled(6, false), 'result[4] = 6');
        assert(*result.data[5] == FixedTrait::new_unscaled(10, false), 'result[5] = 10');
        assert(*result.data[6] == FixedTrait::new_unscaled(1, false), 'result[6] = 1');
        assert(*result.data[7] == FixedTrait::new_unscaled(5, false), 'result[7] = 5');
        assert(*result.data[8] == FixedTrait::new_unscaled(9, false), 'result[8] = 9');
        assert(*result.data[9] == FixedTrait::new_unscaled(3, false), 'result[9] = 3');
        assert(*result.data[10] == FixedTrait::new_unscaled(7, false), 'result[10] = 7');
        assert(*result.data[11] == FixedTrait::new_unscaled(11, false), 'result[11] = 11');
        assert(*result.shape[0] == 2, 'shape[0] = 2');
        assert(*result.shape[1] == 2, 'shape[1] = 2');
        assert(*result.shape[2] == 3, 'shape[2] = 3');

        let mut axes: Array<usize> = ArrayTrait::new();
        axes.append(0);
        axes.append(2);
        axes.append(1);

        let result = tensor.transpose(axes.span());

        assert(*result.data[0] == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert(*result.data[1] == FixedTrait::new_unscaled(2, false), 'result[1] = 2');
        assert(*result.data[2] == FixedTrait::new_unscaled(1, false), 'result[2] = 1');
        assert(*result.data[3] == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
        assert(*result.data[4] == FixedTrait::new_unscaled(4, false), 'result[4] = 4');
        assert(*result.data[5] == FixedTrait::new_unscaled(6, false), 'result[5] = 6');
        assert(*result.data[6] == FixedTrait::new_unscaled(5, false), 'result[6] = 5');
        assert(*result.data[7] == FixedTrait::new_unscaled(7, false), 'result[7] = 7');
        assert(*result.data[8] == FixedTrait::new_unscaled(8, false), 'result[8] = 8');
        assert(*result.data[9] == FixedTrait::new_unscaled(10, false), 'result[9] = 10');
        assert(*result.data[10] == FixedTrait::new_unscaled(9, false), 'result[10] = 9');
        assert(*result.data[11] == FixedTrait::new_unscaled(11, false), 'result[11] = 11');
        assert(*result.shape[0] == 3, 'shape[0] = 3');
        assert(*result.shape[1] == 2, 'shape[1] = 2');
        assert(*result.shape[2] == 2, 'shape[2] = 2');
    }
}
