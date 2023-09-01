use core::serde::Serde;
use core::option::OptionTrait;
use core::clone::Clone;
// ===== 1D ===== //
use orion::numbers::fixed_point::core::FixedTrait;

#[cfg(test)]
mod tensor_1D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;
    use orion::numbers::fixed_point::core::{FixedTrait};
    use orion::numbers::fixed_point::implementations::fp16x16::core::{
        FP16x16Impl, FP16x16PartialEq
    };
    use orion::operators::tensor::implementations::tensor_fp16x16::Tensor_fp16x16;
    use orion::tests::helpers::tensor::fixed_point::fp16x16::{
        fp_tensor_1x3_helper, fp_tensor_2x2_helper, fp_tensor_3x2x2_neg_helper,
        fp_tensor_1x3_neg_helper, fp_tensor_2x2x2_helper
    };
    use orion::operators::tensor::core::TensorTrait;
    use debug::PrintTrait;
    use core::clone::Clone;
    use core::option::OptionTrait;
    use serde::Serde;


    use orion::operators::tensor::implementations::tensor_i8_fp16x16::Tensor_i8_fp16x16;

    // use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;
    // use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{Tensor};


    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};


    // 1D
    fn i8_tensor_1x3_helper() -> Tensor<i8> {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);
        let mut data = ArrayTrait::new();
        data.append(i8 { mag: 0, sign: false });
        data.append(i8 { mag: 1, sign: false });
        data.append(i8 { mag: 2, sign: false });
        
        let tensor = TensorTrait::<i8>::new(sizes.span(), data.span());
        return tensor;
    }

    fn i8_tensor_1x3_neg_helper() -> Tensor<i8> {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);
        let mut data = ArrayTrait::new();
        data.append(i8 { mag: 0, sign: false });
        data.append(i8 { mag: 1, sign: true });
        data.append(i8 { mag: 2, sign: true });
        
        let tensor = TensorTrait::<i8>::new(sizes.span(), data.span());
        return tensor;
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_1x3_last_axis() {
        let tensor = i8_tensor_1x3_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)) == IntegerTrait::new(1, false), 'result[0] = 1');
        assert((*result.data.at(1)) == IntegerTrait::new(0, false), 'result[1] = 0');
        assert((*result.data.at(2)) == IntegerTrait::new(0, false), 'result[2] = 0');
        assert((*result.data.at(3)) == IntegerTrait::new(0, false), 'result[3] = 0');
        assert((*result.data.at(4)) == IntegerTrait::new(1, false), 'result[4] = 1');
        assert((*result.data.at(5)) == IntegerTrait::new(0, false), 'result[5] = 0');
        assert((*result.data.at(6)) == IntegerTrait::new(0, false), 'result[6] = 0');
        assert((*result.data.at(7)) == IntegerTrait::new(0, false), 'result[7] = 0');
        assert((*result.data.at(8)) == IntegerTrait::new(1, false), 'result[8] = 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_1x3_neg_last_axis() {
        let tensor = i8_tensor_1x3_neg_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::None(());

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)) == IntegerTrait::new(1, false), 'result[0] = 1');
        assert((*result.data.at(1)) == IntegerTrait::new(0, false), 'result[1] = 0');
        assert((*result.data.at(2)) == IntegerTrait::new(0, false), 'result[2] = 0');
        assert((*result.data.at(3)) == IntegerTrait::new(0, false), 'result[3] = 0');
        assert((*result.data.at(4)) == IntegerTrait::new(0, false), 'result[4] = 0');
        assert((*result.data.at(5)) == IntegerTrait::new(1, false), 'result[5] = 1');
        assert((*result.data.at(6)) == IntegerTrait::new(0, false), 'result[6] = 0');
        assert((*result.data.at(7)) == IntegerTrait::new(1, false), 'result[7] = 1');
        assert((*result.data.at(8)) == IntegerTrait::new(0, false), 'result[8] = 0');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_1x3_fail() {
        let tensor = i8_tensor_1x3_helper();

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
        let tensor = i8_tensor_1x3_helper();

        let mut values = ArrayTrait::new();
        values.append(0);
        values.append(1);
        let depth = 3;
        let axis: Option<usize> = Option::Some(0);

        let result = tensor.onehot(depth: depth, axis: axis, values: values.span());

        assert((*result.data.at(0)) == IntegerTrait::new(1, false), 'result[0] = 1');
        assert((*result.data.at(1)) == IntegerTrait::new(0, false), 'result[1] = 0');
        assert((*result.data.at(2)) == IntegerTrait::new(0, false), 'result[2] = 0');
        assert((*result.data.at(3)) == IntegerTrait::new(0, false), 'result[3] = 0');
        assert((*result.data.at(4)) == IntegerTrait::new(1, false), 'result[4] = 1');
        assert((*result.data.at(5)) == IntegerTrait::new(0, false), 'result[5] = 0');
        assert((*result.data.at(6)) == IntegerTrait::new(0, false), 'result[6] = 0');
        assert((*result.data.at(7)) == IntegerTrait::new(0, false), 'result[7] = 0');
        assert((*result.data.at(8)) == IntegerTrait::new(1, false), 'result[8] = 1');
    }
}
