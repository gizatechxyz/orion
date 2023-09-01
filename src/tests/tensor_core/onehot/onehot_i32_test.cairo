use core::serde::Serde;
use core::option::OptionTrait;
use core::clone::Clone;
// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;
    use core::option::OptionTrait;
    use serde::Serde;
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
    use orion::operators::tensor::implementations::tensor_i32_fp16x16::Tensor_i32_fp16x16;
    use orion::operators::tensor::core::{TensorTrait, Tensor};

    use orion::tests::helpers::tensor::i32::{
        i32_tensor_1x3_helper, i32_tensor_1x3_neg_helper, i32_tensor_2x2_helper,
        i32_tensor_3x2x2_helper
    };


    fn i32_tensor_3x2x2_new() -> Tensor<i32> {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(i32 { mag: 0, sign: false });
        data.append(i32 { mag: 1, sign: false });
        data.append(i32 { mag: 2, sign: false });
        data.append(i32 { mag: 3, sign: false });
        data.append(i32 { mag: 0, sign: false });
        data.append(i32 { mag: 1, sign: false });
        data.append(i32 { mag: 2, sign: false });
        data.append(i32 { mag: 3, sign: false });
        data.append(i32 { mag: 0, sign: false });
        data.append(i32 { mag: 1, sign: false });
        data.append(i32 { mag: 2, sign: false });
        data.append(i32 { mag: 3, sign: false });

        

        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

        return tensor;
    }

    fn i32_tensor_2x2_pos_neg_new() -> Tensor<i32> {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(i32 { mag: 0, sign: false });
        data.append(i32 { mag: 1, sign: false });
        data.append(i32 { mag: 2, sign: true });
        data.append(i32 { mag: 1, sign: true });

        

        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

        return tensor;
    }


    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_1x3_last_axis() {
        let tensor = i32_tensor_1x3_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)).into() == 1, 'result[0] = 1');
        assert((*result.data.at(1)).into() == 0, 'result[1] = 0');
        assert((*result.data.at(2)).into() == 0, 'result[2] = 0');
        assert((*result.data.at(3)).into() == 0, 'result[3] = 0');
        assert((*result.data.at(4)).into() == 1, 'result[4] = 1');
        assert((*result.data.at(5)).into() == 0, 'result[5] = 0');
        assert((*result.data.at(6)).into() == 0, 'result[6] = 0');
        assert((*result.data.at(7)).into() == 0, 'result[7] = 0');
        assert((*result.data.at(8)).into() == 1, 'result[8] = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_1x3_neg_last_axis() {
        let tensor = i32_tensor_1x3_neg_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)).into() == 1, 'result[0] = 1');
        assert((*result.data.at(1)).into() == 0, 'result[1] = 0');
        assert((*result.data.at(2)).into() == 0, 'result[2] = 0');
        assert((*result.data.at(3)).into() == 0, 'result[3] = 0');
        assert((*result.data.at(4)).into() == 0, 'result[4] = 0');
        assert((*result.data.at(5)).into() == 1, 'result[5] = 1');
        assert((*result.data.at(6)).into() == 0, 'result[6] = 0');
        assert((*result.data.at(7)).into() == 1, 'result[7] = 1');
        assert((*result.data.at(8)).into() == 0, 'result[8] = 0');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_1x3_fail() {
        let tensor = i32_tensor_1x3_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::Some(3);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_1x3_Zero_axis() {
        let tensor = i32_tensor_1x3_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::Some(0);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)).into() == 1, 'result[0] = 1');
        assert((*result.data.at(1)).into() == 0, 'result[1] = 0');
        assert((*result.data.at(2)).into() == 0, 'result[2] = 0');
        assert((*result.data.at(3)).into() == 0, 'result[3] = 0');
        assert((*result.data.at(4)).into() == 1, 'result[4] = 1');
        assert((*result.data.at(5)).into() == 0, 'result[5] = 0');
        assert((*result.data.at(6)).into() == 0, 'result[6] = 0');
        assert((*result.data.at(7)).into() == 0, 'result[7] = 0');
        assert((*result.data.at(8)).into() == 1, 'result[8] = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_2x2_post_neg_last_axis() {
        let tensor = i32_tensor_2x2_pos_neg_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)).into() == 1, 'result[0] = 1');
        assert((*result.data.at(1)).into() == 0, 'result[1] = 0');
        assert((*result.data.at(2)).into() == 0, 'result[2] = 0');
        assert((*result.data.at(3)).into() == 0, 'result[3] = 0');
        assert((*result.data.at(4)).into() == 1, 'result[4] = 1');
        assert((*result.data.at(5)).into() == 0, 'result[5] = 0');
        assert((*result.data.at(6)).into() == 0, 'result[6] = 0');
        assert((*result.data.at(7)).into() == 1, 'result[7] = 1');
        assert((*result.data.at(8)).into() == 0, 'result[8] = 0');
        assert((*result.data.at(9)).into() == 0, 'result[9] = 0');
        assert((*result.data.at(10)).into() == 0, 'result[10] = 0');
        assert((*result.data.at(11)).into() == 1, 'result[11] = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_2x2_helper_last_axis() {
        let tensor = i32_tensor_2x2_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)).into() == 1, 'result[0] = 1');
        assert((*result.data.at(1)).into() == 0, 'result[1] = 0');
        assert((*result.data.at(2)).into() == 0, 'result[2] = 0');
        assert((*result.data.at(3)).into() == 0, 'result[3] = 0');
        assert((*result.data.at(4)).into() == 0, 'result[4] = 0');
        assert((*result.data.at(5)).into() == 1, 'result[5] = 1');
        assert((*result.data.at(6)).into() == 0, 'result[6] = 0');
        assert((*result.data.at(7)).into() == 0, 'result[7] = 0');
        assert((*result.data.at(8)).into() == 0, 'result[8] = 0');
        assert((*result.data.at(9)).into() == 0, 'result[9] = 0');
        assert((*result.data.at(10)).into() == 1, 'result[10] = 1');
        assert((*result.data.at(11)).into() == 0, 'result[11] = 0');
        assert((*result.data.at(12)).into() == 0, 'result[12] = 0');
        assert((*result.data.at(13)).into() == 0, 'result[13] = 0');
        assert((*result.data.at(14)).into() == 0, 'result[14] = 0');
        assert((*result.data.at(15)).into() == 1, 'result[15] = 1');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(2)) == 4, 'shape[0] = 4');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2_fail() {
        let tensor = i32_tensor_2x2_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(3);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_2x2_helper_first_axis() {
        let tensor = i32_tensor_2x2_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(0);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)).into() == 1, 'result[0] = 1');
        assert((*result.data.at(1)).into() == 0, 'result[1] = 0');
        assert((*result.data.at(2)).into() == 0, 'result[2] = 0');
        assert((*result.data.at(3)).into() == 0, 'result[3] = 0');
        assert((*result.data.at(4)).into() == 0, 'result[4] = 0');
        assert((*result.data.at(5)).into() == 1, 'result[5] = 1');
        assert((*result.data.at(6)).into() == 0, 'result[6] = 0');
        assert((*result.data.at(7)).into() == 0, 'result[7] = 0');
        assert((*result.data.at(8)).into() == 0, 'result[8] = 0');
        assert((*result.data.at(9)).into() == 0, 'result[9] = 0');
        assert((*result.data.at(10)).into() == 1, 'result[10] = 1');
        assert((*result.data.at(11)).into() == 0, 'result[11] = 0');
        assert((*result.data.at(12)).into() == 0, 'result[12] = 0');
        assert((*result.data.at(13)).into() == 0, 'result[13] = 0');
        assert((*result.data.at(14)).into() == 0, 'result[14] = 0');
        assert((*result.data.at(15)).into() == 1, 'result[15] = 1');

        assert((*result.shape.at(0)) == 4, 'shape[0] = 4');
        assert((*result.shape.at(1)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(2)) == 2, 'shape[0] = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_2x2_helper_second_axis() {
        let tensor = i32_tensor_2x2_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(1);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)).into() == 1, 'result[0] = 1');
        assert((*result.data.at(1)).into() == 0, 'result[1] = 0');
        assert((*result.data.at(2)).into() == 0, 'result[2] = 0');
        assert((*result.data.at(3)).into() == 1, 'result[3] = 1');
        assert((*result.data.at(4)).into() == 0, 'result[4] = 0');
        assert((*result.data.at(5)).into() == 0, 'result[5] = 0');
        assert((*result.data.at(6)).into() == 0, 'result[6] = 0');
        assert((*result.data.at(7)).into() == 0, 'result[7] = 0');
        assert((*result.data.at(8)).into() == 0, 'result[8] = 0');
        assert((*result.data.at(9)).into() == 0, 'result[9] = 0');
        assert((*result.data.at(10)).into() == 0, 'result[10] = 0');
        assert((*result.data.at(11)).into() == 0, 'result[11] = 0');
        assert((*result.data.at(12)).into() == 1, 'result[12] = 0');
        assert((*result.data.at(13)).into() == 0, 'result[13] = 0');
        assert((*result.data.at(14)).into() == 0, 'result[14] = 0');
        assert((*result.data.at(15)).into() == 1, 'result[15] = 1');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 4, 'shape[0] = 4');
        assert((*result.shape.at(2)) == 2, 'shape[0] = 2');
    }


    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_3x2x2_new_last_axis() {
        let tensor = i32_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)).into() == 1, 'result[0] = 1');
        assert((*result.data.at(1)).into() == 0, 'result[1] = 0');
        assert((*result.data.at(2)).into() == 0, 'result[2] = 0');
        assert((*result.data.at(3)).into() == 0, 'result[3] = 0');
        assert((*result.data.at(4)).into() == 0, 'result[4] = 0');
        assert((*result.data.at(5)).into() == 1, 'result[5] = 1');
        assert((*result.data.at(6)).into() == 0, 'result[6] = 0');
        assert((*result.data.at(7)).into() == 0, 'result[7] = 0');
        assert((*result.data.at(8)).into() == 0, 'result[8] = 0');
        assert((*result.data.at(9)).into() == 0, 'result[9] = 0');
        assert((*result.data.at(10)).into() == 1, 'result[10] = 1');
        assert((*result.data.at(11)).into() == 0, 'result[11] = 0');
        assert((*result.data.at(12)).into() == 0, 'result[12] = 0');
        assert((*result.data.at(13)).into() == 0, 'result[13] = 0');
        assert((*result.data.at(14)).into() == 0, 'result[14] = 0');
        assert((*result.data.at(15)).into() == 1, 'result[15] = 1');
        assert((*result.data.at(16)).into() == 1, 'result[16] = 1');
        assert((*result.data.at(21)).into() == 1, 'result[21] = 1');
        assert((*result.data.at(26)).into() == 1, 'result[26] = 1');
        assert((*result.data.at(31)).into() == 1, 'result[31] = 1');
        assert((*result.data.at(32)).into() == 1, 'result[32] = 1');
        assert((*result.data.at(37)).into() == 1, 'result[37] = 1');
        assert((*result.data.at(42)).into() == 1, 'result[42] = 1');
        assert((*result.data.at(46)).into() == 0, 'result[46] = 0');
        assert((*result.data.at(47)).into() == 1, 'result[47] = 1');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 2, 'shape[2] = 2');
        assert((*result.shape.at(3)) == 4, 'shape[0] = 4');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2_fail() {
        let tensor = i32_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(4);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_3x2x2_new_first_axis() {
        let tensor = i32_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(2);
        values.append(5);
        let depth = 4;
        let axis: Option<usize> = Option::Some(0);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)).into() == 5, 'result[0] = 5');
        assert((*result.data.at(1)).into() == 2, 'result[1] = 2');
        assert((*result.data.at(2)).into() == 2, 'result[2] = 2');
        assert((*result.data.at(3)).into() == 2, 'result[3] = 2');
        assert((*result.data.at(4)).into() == 5, 'result[4] = 5');
        assert((*result.data.at(5)).into() == 2, 'result[5] = 2');
        assert((*result.data.at(6)).into() == 2, 'result[6] = 2');
        assert((*result.data.at(7)).into() == 2, 'result[7] = 2');
        assert((*result.data.at(8)).into() == 5, 'result[8] = 5');
        assert((*result.data.at(9)).into() == 2, 'result[9] = 2');
        assert((*result.data.at(10)).into() == 2, 'result[10] = 2');
        assert((*result.data.at(11)).into() == 2, 'result[11] = 2');
        assert((*result.data.at(12)).into() == 2, 'result[12] = 2');
        assert((*result.data.at(13)).into() == 5, 'result[13] = 5');
        assert((*result.data.at(14)).into() == 2, 'result[14] = 2');
        assert((*result.data.at(17)).into() == 5, 'result[17] = 5');
        assert((*result.data.at(21)).into() == 5, 'result[21] = 5');
        assert((*result.data.at(26)).into() == 5, 'result[26] = 5');
        assert((*result.data.at(30)).into() == 5, 'result[30] = 5');
        assert((*result.data.at(34)).into() == 5, 'result[34] = 5');
        assert((*result.data.at(39)).into() == 5, 'result[39] = 5');
        assert((*result.data.at(43)).into() == 5, 'result[43] = 5');
        assert((*result.data.at(46)).into() == 2, 'result[46] = 2');
        assert((*result.data.at(47)).into() == 5, 'result[47] = 5');

        assert((*result.shape.at(0)) == 4, 'shape[0] = 4');
        assert((*result.shape.at(1)) == 3, 'shape[1] = 3');
        assert((*result.shape.at(2)) == 2, 'shape[2] = 3');
        assert((*result.shape.at(3)) == 2, 'shape[0] = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_3x2x2_new_second_axis() {
        let tensor = i32_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(1);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)).into() == 1, 'result[0] = 1');
        assert((*result.data.at(1)).into() == 0, 'result[1] = 0');
        assert((*result.data.at(2)).into() == 0, 'result[2] = 0');
        assert((*result.data.at(3)).into() == 0, 'result[3] = 0');
        assert((*result.data.at(4)).into() == 0, 'result[4] = 0');
        assert((*result.data.at(5)).into() == 1, 'result[5] = 1');
        assert((*result.data.at(6)).into() == 0, 'result[6] = 0');
        assert((*result.data.at(7)).into() == 0, 'result[7] = 0');
        assert((*result.data.at(8)).into() == 0, 'result[8] = 0');
        assert((*result.data.at(9)).into() == 0, 'result[9] = 0');
        assert((*result.data.at(10)).into() == 1, 'result[10] = 1');
        assert((*result.data.at(11)).into() == 0, 'result[11] = 0');
        assert((*result.data.at(12)).into() == 0, 'result[12] = 0');
        assert((*result.data.at(13)).into() == 0, 'result[13] = 0');
        assert((*result.data.at(14)).into() == 0, 'result[14] = 0');
        assert((*result.data.at(15)).into() == 1, 'result[15] = 1');
        assert((*result.data.at(16)).into() == 1, 'result[16] = 1');
        assert((*result.data.at(21)).into() == 1, 'result[21] = 1');
        assert((*result.data.at(26)).into() == 1, 'result[26] = 1');
        assert((*result.data.at(31)).into() == 1, 'result[31] = 1');
        assert((*result.data.at(32)).into() == 1, 'result[32] = 1');
        assert((*result.data.at(37)).into() == 1, 'result[37] = 1');
        assert((*result.data.at(42)).into() == 1, 'result[42] = 1');
        assert((*result.data.at(46)).into() == 0, 'result[46] = 0');
        assert((*result.data.at(47)).into() == 1, 'result[47] = 1');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 4, 'shape[1] = 4');
        assert((*result.shape.at(2)) == 2, 'shape[2] = 3');
        assert((*result.shape.at(3)) == 2, 'shape[0] = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_3x2x2_new_third_axis() {
        let tensor = i32_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(2);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)).into() == 1, 'result[0] = 1');
        assert((*result.data.at(1)).into() == 0, 'result[1] = 0');
        assert((*result.data.at(2)).into() == 0, 'result[2] = 0');
        assert((*result.data.at(3)).into() == 1, 'result[3] = 1');
        assert((*result.data.at(4)).into() == 0, 'result[4] = 0');
        assert((*result.data.at(5)).into() == 0, 'result[5] = 0');
        assert((*result.data.at(6)).into() == 0, 'result[6] = 0');
        assert((*result.data.at(12)).into() == 1, 'result[12] = 1');
        assert((*result.data.at(13)).into() == 0, 'result[13] = 0');
        assert((*result.data.at(14)).into() == 0, 'result[14] = 0');
        assert((*result.data.at(15)).into() == 1, 'result[15] = 1');
        assert((*result.data.at(16)).into() == 1, 'result[16] = 1');
        assert((*result.data.at(19)).into() == 1, 'result[19] = 1');
        assert((*result.data.at(21)).into() == 0, 'result[21] = 0');
        assert((*result.data.at(26)).into() == 0, 'result[26] = 0');
        assert((*result.data.at(28)).into() == 1, 'result[28] = 1');
        assert((*result.data.at(31)).into() == 1, 'result[31] = 1');
        assert((*result.data.at(32)).into() == 1, 'result[32] = 1');
        assert((*result.data.at(35)).into() == 1, 'result[35] = 1');
        assert((*result.data.at(37)).into() == 0, 'result[37] = 0');
        assert((*result.data.at(44)).into() == 1, 'result[44] = 1');
        assert((*result.data.at(46)).into() == 0, 'result[46] = 0');
        assert((*result.data.at(47)).into() == 1, 'result[47] = 1');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 4, 'shape[2] = 4');
        assert((*result.shape.at(3)) == 2, 'shape[0] = 2');
    }
}
