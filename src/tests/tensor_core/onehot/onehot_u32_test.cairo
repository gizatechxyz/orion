use core::serde::Serde;
use core::option::OptionTrait;
use core::clone::Clone;
// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;

    use orion::operators::tensor::implementations::tensor_u32_fp16x16::Tensor_u32_fp16x16;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams, Tensor};
    use orion::tests::helpers::tensor::u32::{u32_tensor_1x3_helper, u32_tensor_2x2_helper};


    fn u32_tensor_3x2x2_new() -> Tensor<u32> {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(0);
        data.append(1);
        data.append(2);
        data.append(3);
        data.append(0);
        data.append(1);
        data.append(2);
        data.append(3);
        data.append(0);
        data.append(1);
        data.append(2);
        data.append(3);

        let extra = Option::<ExtraParams>::None(());

        let tensor = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

        return tensor;
    }


    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_1x3_last_axis() {
        let tensor = u32_tensor_1x3_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == 1, 'result[0] = 1');
        assert((*result.data[1]) == 0, 'result[1] = 0');
        assert((*result.data[2]) == 0, 'result[2] = 0');
        assert((*result.data[3]) == 0, 'result[3] = 0');
        assert((*result.data[4]) == 1, 'result[4] = 1');
        assert((*result.data[5]) == 0, 'result[5] = 0');
        assert((*result.data[6]) == 0, 'result[6] = 0');
        assert((*result.data[7]) == 0, 'result[7] = 0');
        assert((*result.data[8]) == 1, 'result[8] = 1');
    }

    #[test]
    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_1x3_fail() {
        let tensor = u32_tensor_1x3_helper();

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
        let tensor = u32_tensor_1x3_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::Some(0);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == 1, 'result[0] = 1');
        assert((*result.data[1]) == 0, 'result[1] = 0');
        assert((*result.data[2]) == 0, 'result[2] = 0');
        assert((*result.data[3]) == 0, 'result[3] = 0');
        assert((*result.data[4]) == 1, 'result[4] = 1');
        assert((*result.data[5]) == 0, 'result[5] = 0');
        assert((*result.data[6]) == 0, 'result[6] = 0');
        assert((*result.data[7]) == 0, 'result[7] = 0');
        assert((*result.data[8]) == 1, 'result[8] = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_1x3_axis_one() {
        let tensor = u32_tensor_1x3_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::Some(1);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == 1, 'result[0] = 1');
        assert((*result.data[1]) == 0, 'result[1] = 0');
        assert((*result.data[2]) == 0, 'result[2] = 0');
        assert((*result.data[3]) == 0, 'result[3] = 0');
        assert((*result.data[4]) == 1, 'result[4] = 1');
        assert((*result.data[5]) == 0, 'result[5] = 0');
        assert((*result.data[6]) == 0, 'result[6] = 0');
        assert((*result.data[7]) == 0, 'result[7] = 0');
        assert((*result.data[8]) == 1, 'result[8] = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_2x2_helper_last_axis() {
        let tensor = u32_tensor_2x2_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == 1, 'result[0] = 1');
        assert((*result.data[1]) == 0, 'result[1] = 0');
        assert((*result.data[2]) == 0, 'result[2] = 0');
        assert((*result.data[3]) == 0, 'result[3] = 0');
        assert((*result.data[4]) == 0, 'result[4] = 0');
        assert((*result.data[5]) == 1, 'result[5] = 1');
        assert((*result.data[6]) == 0, 'result[6] = 0');
        assert((*result.data[7]) == 0, 'result[7] = 0');
        assert((*result.data[8]) == 0, 'result[8] = 0');
        assert((*result.data[9]) == 0, 'result[9] = 0');
        assert((*result.data[10]) == 1, 'result[10] = 1');
        assert((*result.data[11]) == 0, 'result[11] = 0');
        assert((*result.data[12]) == 0, 'result[12] = 0');
        assert((*result.data[13]) == 0, 'result[13] = 0');
        assert((*result.data[14]) == 0, 'result[14] = 0');
        assert((*result.data[15]) == 1, 'result[15] = 1');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(2)) == 4, 'shape[0] = 4');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2_fail() {
        let tensor = u32_tensor_2x2_helper();

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
        let tensor = u32_tensor_2x2_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(0);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == 1, 'result[0] = 1');
        assert((*result.data[1]) == 0, 'result[1] = 0');
        assert((*result.data[2]) == 0, 'result[2] = 0');
        assert((*result.data[3]) == 0, 'result[3] = 0');
        assert((*result.data[4]) == 0, 'result[4] = 0');
        assert((*result.data[5]) == 1, 'result[5] = 1');
        assert((*result.data[6]) == 0, 'result[6] = 0');
        assert((*result.data[7]) == 0, 'result[7] = 0');
        assert((*result.data[8]) == 0, 'result[8] = 0');
        assert((*result.data[9]) == 0, 'result[9] = 0');
        assert((*result.data[10]) == 1, 'result[10] = 1');
        assert((*result.data[11]) == 0, 'result[11] = 0');
        assert((*result.data[12]) == 0, 'result[12] = 0');
        assert((*result.data[13]) == 0, 'result[13] = 0');
        assert((*result.data[14]) == 0, 'result[14] = 0');
        assert((*result.data[15]) == 1, 'result[15] = 1');

        assert((*result.shape.at(0)) == 4, 'shape[0] = 4');
        assert((*result.shape.at(1)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(2)) == 2, 'shape[0] = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_2x2_helper_second_axis() {
        let tensor = u32_tensor_2x2_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(1);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == 1, 'result[0] = 1');
        assert((*result.data[1]) == 0, 'result[1] = 0');
        assert((*result.data[2]) == 0, 'result[2] = 0');
        assert((*result.data[3]) == 1, 'result[3] = 1');
        assert((*result.data[4]) == 0, 'result[4] = 0');
        assert((*result.data[5]) == 0, 'result[5] = 0');
        assert((*result.data[6]) == 0, 'result[6] = 0');
        assert((*result.data[7]) == 0, 'result[7] = 0');
        assert((*result.data[8]) == 0, 'result[8] = 0');
        assert((*result.data[9]) == 0, 'result[9] = 0');
        assert((*result.data[10]) == 0, 'result[10] = 1');
        assert((*result.data[11]) == 0, 'result[11] = 0');
        assert((*result.data[12]) == 1, 'result[12] = 1');
        assert((*result.data[13]) == 0, 'result[13] = 0');
        assert((*result.data[14]) == 0, 'result[14] = 0');
        assert((*result.data[15]) == 1, 'result[15] = 1');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 4, 'shape[0] = 4');
        assert((*result.shape.at(2)) == 2, 'shape[0] = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_3x2x2_new_last_axis() {
        let tensor = u32_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::None(());
        // let axis: Option<usize> = Option::Some(3);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == 1, 'result[0] = 1');
        assert((*result.data[1]) == 0, 'result[1] = 0');
        assert((*result.data[2]) == 0, 'result[2] = 0');
        assert((*result.data[3]) == 0, 'result[3] = 0');
        assert((*result.data[4]) == 0, 'result[4] = 0');
        assert((*result.data[5]) == 1, 'result[5] = 1');
        assert((*result.data[6]) == 0, 'result[6] = 0');
        assert((*result.data[7]) == 0, 'result[7] = 0');
        assert((*result.data[8]) == 0, 'result[8] = 0');
        assert((*result.data[9]) == 0, 'result[9] = 0');
        assert((*result.data[10]) == 1, 'result[10] = 1');
        assert((*result.data[11]) == 0, 'result[11] = 0');
        assert((*result.data[12]) == 0, 'result[12] = 0');
        assert((*result.data[13]) == 0, 'result[13] = 0');
        assert((*result.data[14]) == 0, 'result[14] = 0');
        assert((*result.data[15]) == 1, 'result[15] = 1');
        assert((*result.data[16]) == 1, 'result[16] = 1');
        assert((*result.data[21]) == 1, 'result[21] = 1');
        assert((*result.data[26]) == 1, 'result[26] = 1');
        assert((*result.data[31]) == 1, 'result[31] = 1');
        assert((*result.data[32]) == 1, 'result[32] = 1');
        assert((*result.data[37]) == 1, 'result[37] = 1');
        assert((*result.data[42]) == 1, 'result[42] = 1');
        assert((*result.data[46]) == 0, 'result[46] = 0');
        assert((*result.data[47]) == 1, 'result[47] = 1');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 2, 'shape[2] = 2');
        assert((*result.shape.at(3)) == 4, 'shape[0] = 4');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2_fail() {
        let tensor = u32_tensor_3x2x2_new();

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
        let tensor = u32_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(2);
        values.append(5);
        let depth = 4;
        let axis: Option<usize> = Option::Some(0);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == 5, 'result[0] = 5');
        assert((*result.data[1]) == 2, 'result[1] = 2');
        assert((*result.data[2]) == 2, 'result[2] = 2');
        assert((*result.data[3]) == 2, 'result[3] = 2');
        assert((*result.data[4]) == 5, 'result[4] = 5');
        assert((*result.data[5]) == 2, 'result[5] = 2');
        assert((*result.data[6]) == 2, 'result[6] = 2');
        assert((*result.data[7]) == 2, 'result[7] = 2');
        assert((*result.data[8]) == 5, 'result[8] = 5');
        assert((*result.data[9]) == 2, 'result[9] = 2');
        assert((*result.data[10]) == 2, 'result[10] = 2');
        assert((*result.data[11]) == 2, 'result[11] = 2');
        assert((*result.data[12]) == 2, 'result[12] = 2');
        assert((*result.data[13]) == 5, 'result[13] = 5');
        assert((*result.data[14]) == 2, 'result[14] = 2');
        assert((*result.data[17]) == 5, 'result[17] = 5');
        assert((*result.data[21]) == 5, 'result[21] = 5');
        assert((*result.data[26]) == 5, 'result[26] = 5');
        assert((*result.data[30]) == 5, 'result[30] = 5');
        assert((*result.data[34]) == 5, 'result[34] = 5');
        assert((*result.data[39]) == 5, 'result[39] = 5');
        assert((*result.data[43]) == 5, 'result[43] = 5');
        assert((*result.data[46]) == 2, 'result[46] = 2');
        assert((*result.data[47]) == 5, 'result[47] = 5');

        assert((*result.shape.at(0)) == 4, 'shape[0] = 4');
        assert((*result.shape.at(1)) == 3, 'shape[1] = 3');
        assert((*result.shape.at(2)) == 2, 'shape[2] = 3');
        assert((*result.shape.at(3)) == 2, 'shape[0] = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_3x2x2_new_second_axis() {
        let tensor = u32_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(1);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == 1, 'result[0] = 1');
        assert((*result.data[1]) == 0, 'result[1] = 0');
        assert((*result.data[2]) == 0, 'result[2] = 0');
        assert((*result.data[3]) == 0, 'result[3] = 0');
        assert((*result.data[4]) == 0, 'result[4] = 0');
        assert((*result.data[5]) == 1, 'result[5] = 1');
        assert((*result.data[6]) == 0, 'result[6] = 0');
        assert((*result.data[7]) == 0, 'result[7] = 0');
        assert((*result.data[8]) == 0, 'result[8] = 0');
        assert((*result.data[9]) == 0, 'result[9] = 0');
        assert((*result.data[10]) == 1, 'result[10] = 1');
        assert((*result.data[11]) == 0, 'result[11] = 0');
        assert((*result.data[12]) == 0, 'result[12] = 0');
        assert((*result.data[13]) == 0, 'result[13] = 0');
        assert((*result.data[14]) == 0, 'result[14] = 0');
        assert((*result.data[15]) == 1, 'result[15] = 1');
        assert((*result.data[16]) == 1, 'result[16] = 1');
        assert((*result.data[21]) == 1, 'result[21] = 1');
        assert((*result.data[26]) == 1, 'result[26] = 1');
        assert((*result.data[31]) == 1, 'result[31] = 1');
        assert((*result.data[32]) == 1, 'result[32] = 1');
        assert((*result.data[37]) == 1, 'result[37] = 1');
        assert((*result.data[42]) == 1, 'result[42] = 1');
        assert((*result.data[46]) == 0, 'result[46] = 0');
        assert((*result.data[47]) == 1, 'result[47] = 1');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 4, 'shape[1] = 4');
        assert((*result.shape.at(2)) == 2, 'shape[2] = 3');
        assert((*result.shape.at(3)) == 2, 'shape[0] = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_3x2x2_new_third_axis() {
        let tensor = u32_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(2);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == 1, 'result[0] = 1');
        assert((*result.data[1]) == 0, 'result[1] = 0');
        assert((*result.data[2]) == 0, 'result[2] = 0');
        assert((*result.data[3]) == 1, 'result[3] = 1');
        assert((*result.data[4]) == 0, 'result[4] = 0');
        assert((*result.data[5]) == 0, 'result[5] = 0');
        assert((*result.data[6]) == 0, 'result[6] = 0');
        assert((*result.data[12]) == 1, 'result[12] = 1');
        assert((*result.data[13]) == 0, 'result[13] = 0');
        assert((*result.data[14]) == 0, 'result[14] = 0');
        assert((*result.data[15]) == 1, 'result[15] = 1');
        assert((*result.data[16]) == 1, 'result[16] = 1');
        assert((*result.data[19]) == 1, 'result[19] = 1');
        assert((*result.data[21]) == 0, 'result[21] = 0');
        assert((*result.data[26]) == 0, 'result[26] = 0');
        assert((*result.data[28]) == 1, 'result[28] = 1');
        assert((*result.data[31]) == 1, 'result[31] = 1');
        assert((*result.data[32]) == 1, 'result[32] = 1');
        assert((*result.data[35]) == 1, 'result[35] = 1');
        assert((*result.data[37]) == 0, 'result[37] = 0');
        assert((*result.data[44]) == 1, 'result[44] = 1');
        assert((*result.data[46]) == 0, 'result[46] = 0');
        assert((*result.data[47]) == 1, 'result[47] = 1');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 4, 'shape[2] = 4');
        assert((*result.shape.at(3)) == 2, 'shape[0] = 2');
    }
}
