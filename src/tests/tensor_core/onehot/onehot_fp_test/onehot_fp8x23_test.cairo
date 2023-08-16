use core::serde::Serde;
use core::option::OptionTrait;
use core::clone::Clone;
// ===== 1D ===== //
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};

#[cfg(test)]
mod tensor_1D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
    use orion::numbers::fixed_point::implementations::fp8x23::core::{FP8x23Impl, FP8x23PartialEq};
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::TensorTrait;
    use orion::tests::helpers::tensor::fixed_point::fp8x23::{
        fp_tensor_1x3_helper, fp_tensor_2x2_helper, fp_tensor_3x2x2_neg_helper,
        fp_tensor_1x3_neg_helper, fp_tensor_2x2x2_helper
    };
    use debug::PrintTrait;
    use core::clone::Clone;
    use core::option::OptionTrait;
    use serde::Serde;

    use orion::operators::tensor::core::{Tensor, ExtraParams};

    fn fp_tensor_3x2x2_new() -> Tensor<FixedType> {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));

        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

        return tensor;
    }

    fn fp_tensor_2x2_pos_neg_new() -> Tensor<FixedType> {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, true));
        data.append(FixedTrait::new_unscaled(1, true));

        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

        return tensor;
    }


    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_1x3_last_axis() {
        let tensor = fp_tensor_1x3_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(1, false), 'result[4] = 1');
        assert((*result.data[5]) == FixedTrait::new_unscaled(0, false), 'result[5] = 0');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(0, false), 'result[7] = 0');
        assert((*result.data[8]) == FixedTrait::new_unscaled(1, false), 'result[8] = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_1x3_neg_last_axis() {
        let tensor = fp_tensor_1x3_neg_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0, false), 'result[4] = 1');
        assert((*result.data[5]) == FixedTrait::new_unscaled(1, false), 'result[5] = 0');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(1, false), 'result[7] = 0');
        assert((*result.data[8]) == FixedTrait::new_unscaled(0, false), 'result[8] = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_2x2_post_neg_last_axis() {
        let tensor = fp_tensor_2x2_pos_neg_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(1, false), 'result[4] = 1');
        assert((*result.data[5]) == FixedTrait::new_unscaled(0, false), 'result[5] = 0');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(1, false), 'result[7] = 1');
        assert((*result.data[8]) == FixedTrait::new_unscaled(0, false), 'result[8] = 0');
        assert((*result.data[9]) == FixedTrait::new_unscaled(0, false), 'result[9] = 0');
        assert((*result.data[10]) == FixedTrait::new_unscaled(0, false), 'result[10] = 0');
        assert((*result.data[11]) == FixedTrait::new_unscaled(1, false), 'result[11] = 0');
    }


    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_1x3_fail() {
        let tensor = fp_tensor_1x3_helper();

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
        let tensor = fp_tensor_1x3_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::Some(0);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(1, false), 'result[4] = 1');
        assert((*result.data[5]) == FixedTrait::new_unscaled(0, false), 'result[5] = 0');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(0, false), 'result[7] = 0');
        assert((*result.data[8]) == FixedTrait::new_unscaled(1, false), 'result[8] = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_1x3_axis_one() {
        let tensor = fp_tensor_1x3_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::Some(1);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(1, false), 'result[4] = 1');
        assert((*result.data[5]) == FixedTrait::new_unscaled(0, false), 'result[5] = 0');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(0, false), 'result[7] = 0');
        assert((*result.data[8]) == FixedTrait::new_unscaled(1, false), 'result[8] = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_2x2_helper_last_axis() {
        let tensor = fp_tensor_2x2_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0, false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(1, false), 'result[5] = 1');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(0, false), 'result[7] = 0');
        assert((*result.data[8]) == FixedTrait::new_unscaled(0, false), 'result[8] = 0');
        assert((*result.data[9]) == FixedTrait::new_unscaled(0, false), 'result[9] = 0');
        assert((*result.data[10]) == FixedTrait::new_unscaled(1, false), 'result[10] = 1');
        assert((*result.data[11]) == FixedTrait::new_unscaled(0, false), 'result[11] = 0');
        assert((*result.data[12]) == FixedTrait::new_unscaled(0, false), 'result[12] = 0');
        assert((*result.data[13]) == FixedTrait::new_unscaled(0, false), 'result[13] = 0');
        assert((*result.data[14]) == FixedTrait::new_unscaled(0, false), 'result[14] = 0');
        assert((*result.data[15]) == FixedTrait::new_unscaled(1, false), 'result[15] = 1');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(2)) == 4, 'shape[0] = 4');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2_fail() {
        let tensor = fp_tensor_2x2_helper();

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
        let tensor = fp_tensor_2x2_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(0);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0, false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(1, false), 'result[5] = 1');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(0, false), 'result[7] = 0');
        assert((*result.data[8]) == FixedTrait::new_unscaled(0, false), 'result[8] = 0');
        assert((*result.data[9]) == FixedTrait::new_unscaled(0, false), 'result[9] = 0');
        assert((*result.data[10]) == FixedTrait::new_unscaled(1, false), 'result[10] = 1');
        assert((*result.data[11]) == FixedTrait::new_unscaled(0, false), 'result[11] = 0');
        assert((*result.data[12]) == FixedTrait::new_unscaled(0, false), 'result[12] = 0');
        assert((*result.data[13]) == FixedTrait::new_unscaled(0, false), 'result[13] = 0');
        assert((*result.data[14]) == FixedTrait::new_unscaled(0, false), 'result[14] = 0');
        assert((*result.data[15]) == FixedTrait::new_unscaled(1, false), 'result[15] = 1');

        assert((*result.shape.at(0)) == 4, 'shape[0] = 4');
        assert((*result.shape.at(1)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(2)) == 2, 'shape[0] = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_2x2_helper_second_axis() {
        let tensor = fp_tensor_2x2_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(1);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(1, false), 'result[3] = 1');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0, false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(0, false), 'result[5] = 0');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(0, false), 'result[7] = 0');
        assert((*result.data[8]) == FixedTrait::new_unscaled(0, false), 'result[8] = 0');
        assert((*result.data[9]) == FixedTrait::new_unscaled(0, false), 'result[9] = 0');
        assert((*result.data[10]) == FixedTrait::new_unscaled(0, false), 'result[10] = 1');
        assert((*result.data[11]) == FixedTrait::new_unscaled(0, false), 'result[11] = 0');
        assert((*result.data[12]) == FixedTrait::new_unscaled(1, false), 'result[12] = 1');
        assert((*result.data[13]) == FixedTrait::new_unscaled(0, false), 'result[13] = 0');
        assert((*result.data[14]) == FixedTrait::new_unscaled(0, false), 'result[14] = 0');
        assert((*result.data[15]) == FixedTrait::new_unscaled(1, false), 'result[15] = 1');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 4, 'shape[0] = 4');
        assert((*result.shape.at(2)) == 2, 'shape[0] = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_3x2x2_new_last_axis() {
        let tensor = fp_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::None(());
        // let axis: Option<usize> = Option::Some(3);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0, false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(1, false), 'result[5] = 1');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(0, false), 'result[7] = 0');
        assert((*result.data[8]) == FixedTrait::new_unscaled(0, false), 'result[8] = 0');
        assert((*result.data[9]) == FixedTrait::new_unscaled(0, false), 'result[9] = 0');
        assert((*result.data[10]) == FixedTrait::new_unscaled(1, false), 'result[10] = 1');
        assert((*result.data[11]) == FixedTrait::new_unscaled(0, false), 'result[11] = 0');
        assert((*result.data[12]) == FixedTrait::new_unscaled(0, false), 'result[12] = 0');
        assert((*result.data[13]) == FixedTrait::new_unscaled(0, false), 'result[13] = 0');
        assert((*result.data[14]) == FixedTrait::new_unscaled(0, false), 'result[14] = 0');
        assert((*result.data[15]) == FixedTrait::new_unscaled(1, false), 'result[15] = 1');
        assert((*result.data[16]) == FixedTrait::new_unscaled(1, false), 'result[16] = 1');
        assert((*result.data[21]) == FixedTrait::new_unscaled(1, false), 'result[21] = 1');
        assert((*result.data[26]) == FixedTrait::new_unscaled(1, false), 'result[26] = 1');
        assert((*result.data[31]) == FixedTrait::new_unscaled(1, false), 'result[31] = 1');
        assert((*result.data[32]) == FixedTrait::new_unscaled(1, false), 'result[32] = 1');
        assert((*result.data[37]) == FixedTrait::new_unscaled(1, false), 'result[37] = 1');
        assert((*result.data[42]) == FixedTrait::new_unscaled(1, false), 'result[42] = 1');
        assert((*result.data[46]) == FixedTrait::new_unscaled(0, false), 'result[46] = 0');
        assert((*result.data[47]) == FixedTrait::new_unscaled(1, false), 'result[47] = 1');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 2, 'shape[2] = 2');
        assert((*result.shape.at(3)) == 4, 'shape[0] = 4');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2_fail() {
        let tensor = fp_tensor_3x2x2_new();

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
        let tensor = fp_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(2);
        values.append(5);
        let depth = 4;
        let axis: Option<usize> = Option::Some(0);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(5, false), 'result[0] = 5');
        assert((*result.data[1]) == FixedTrait::new_unscaled(2, false), 'result[1] = 2');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(2, false), 'result[3] = 2');
        assert((*result.data[4]) == FixedTrait::new_unscaled(5, false), 'result[4] = 5');
        assert((*result.data[5]) == FixedTrait::new_unscaled(2, false), 'result[5] = 2');
        assert((*result.data[6]) == FixedTrait::new_unscaled(2, false), 'result[6] = 2');
        assert((*result.data[7]) == FixedTrait::new_unscaled(2, false), 'result[7] = 2');
        assert((*result.data[8]) == FixedTrait::new_unscaled(5, false), 'result[8] = 5');
        assert((*result.data[9]) == FixedTrait::new_unscaled(2, false), 'result[9] = 2');
        assert((*result.data[10]) == FixedTrait::new_unscaled(2, false), 'result[10] = 2');
        assert((*result.data[11]) == FixedTrait::new_unscaled(2, false), 'result[11] = 2');
        assert((*result.data[12]) == FixedTrait::new_unscaled(2, false), 'result[12] = 2');
        assert((*result.data[13]) == FixedTrait::new_unscaled(5, false), 'result[13] = 5');
        assert((*result.data[14]) == FixedTrait::new_unscaled(2, false), 'result[14] = 2');
        assert((*result.data[17]) == FixedTrait::new_unscaled(5, false), 'result[17] = 5');
        assert((*result.data[21]) == FixedTrait::new_unscaled(5, false), 'result[21] = 5');
        assert((*result.data[26]) == FixedTrait::new_unscaled(5, false), 'result[26] = 5');
        assert((*result.data[30]) == FixedTrait::new_unscaled(5, false), 'result[30] = 5');
        assert((*result.data[34]) == FixedTrait::new_unscaled(5, false), 'result[34] = 5');
        assert((*result.data[39]) == FixedTrait::new_unscaled(5, false), 'result[39] = 5');
        assert((*result.data[43]) == FixedTrait::new_unscaled(5, false), 'result[43] = 5');
        assert((*result.data[46]) == FixedTrait::new_unscaled(2, false), 'result[46] = 2');
        assert((*result.data[47]) == FixedTrait::new_unscaled(5, false), 'result[47] = 5');

        assert((*result.shape.at(0)) == 4, 'shape[0] = 4');
        assert((*result.shape.at(1)) == 3, 'shape[1] = 3');
        assert((*result.shape.at(2)) == 2, 'shape[2] = 3');
        assert((*result.shape.at(3)) == 2, 'shape[0] = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_3x2x2_new_second_axis() {
        let tensor = fp_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(1);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0, false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(1, false), 'result[5] = 1');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(0, false), 'result[7] = 0');
        assert((*result.data[8]) == FixedTrait::new_unscaled(0, false), 'result[8] = 0');
        assert((*result.data[9]) == FixedTrait::new_unscaled(0, false), 'result[9] = 0');
        assert((*result.data[10]) == FixedTrait::new_unscaled(1, false), 'result[10] = 1');
        assert((*result.data[11]) == FixedTrait::new_unscaled(0, false), 'result[11] = 0');
        assert((*result.data[12]) == FixedTrait::new_unscaled(0, false), 'result[12] = 0');
        assert((*result.data[13]) == FixedTrait::new_unscaled(0, false), 'result[13] = 0');
        assert((*result.data[14]) == FixedTrait::new_unscaled(0, false), 'result[14] = 0');
        assert((*result.data[15]) == FixedTrait::new_unscaled(1, false), 'result[15] = 1');
        assert((*result.data[16]) == FixedTrait::new_unscaled(1, false), 'result[16] = 1');
        assert((*result.data[21]) == FixedTrait::new_unscaled(1, false), 'result[21] = 1');
        assert((*result.data[26]) == FixedTrait::new_unscaled(1, false), 'result[26] = 1');
        assert((*result.data[31]) == FixedTrait::new_unscaled(1, false), 'result[31] = 1');
        assert((*result.data[32]) == FixedTrait::new_unscaled(1, false), 'result[32] = 1');
        assert((*result.data[37]) == FixedTrait::new_unscaled(1, false), 'result[37] = 1');
        assert((*result.data[42]) == FixedTrait::new_unscaled(1, false), 'result[42] = 1');
        assert((*result.data[46]) == FixedTrait::new_unscaled(0, false), 'result[46] = 0');
        assert((*result.data[47]) == FixedTrait::new_unscaled(1, false), 'result[47] = 1');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 4, 'shape[1] = 4');
        assert((*result.shape.at(2)) == 2, 'shape[2] = 3');
        assert((*result.shape.at(3)) == 2, 'shape[0] = 2');
    }

    #[test]
    #[available_gas(20000000)]
    fn fp_tensor_onehot_3x2x2_new_third_axis() {
        let tensor = fp_tensor_3x2x2_new();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 4;
        let axis: Option<usize> = Option::Some(2);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data[0]) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(1, false), 'result[3] = 1');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0, false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(0, false), 'result[5] = 0');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[12]) == FixedTrait::new_unscaled(1, false), 'result[12] = 1');
        assert((*result.data[13]) == FixedTrait::new_unscaled(0, false), 'result[13] = 0');
        assert((*result.data[14]) == FixedTrait::new_unscaled(0, false), 'result[14] = 0');
        assert((*result.data[15]) == FixedTrait::new_unscaled(1, false), 'result[15] = 1');
        assert((*result.data[16]) == FixedTrait::new_unscaled(1, false), 'result[16] = 1');
        assert((*result.data[19]) == FixedTrait::new_unscaled(1, false), 'result[19] = 1');
        assert((*result.data[21]) == FixedTrait::new_unscaled(0, false), 'result[21] = 0');
        assert((*result.data[26]) == FixedTrait::new_unscaled(0, false), 'result[26] = 0');
        assert((*result.data[28]) == FixedTrait::new_unscaled(1, false), 'result[28] = 1');
        assert((*result.data[31]) == FixedTrait::new_unscaled(1, false), 'result[31] = 1');
        assert((*result.data[32]) == FixedTrait::new_unscaled(1, false), 'result[32] = 1');
        assert((*result.data[35]) == FixedTrait::new_unscaled(1, false), 'result[35] = 1');
        assert((*result.data[37]) == FixedTrait::new_unscaled(0, false), 'result[37] = 0');
        assert((*result.data[44]) == FixedTrait::new_unscaled(1, false), 'result[44] = 1');
        assert((*result.data[46]) == FixedTrait::new_unscaled(0, false), 'result[46] = 0');
        assert((*result.data[47]) == FixedTrait::new_unscaled(1, false), 'result[47] = 1');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 4, 'shape[2] = 4');
        assert((*result.shape.at(3)) == 2, 'shape[0] = 2');
    }
}

