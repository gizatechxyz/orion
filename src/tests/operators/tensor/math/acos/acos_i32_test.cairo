// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    use orion::tests::helpers::tensor::i32::i32_tensor_1x3_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};

    fn i32_tensor_1x3_helper_in_test() -> Tensor<i32> {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);

        let mut data = ArrayTrait::new();
        data.append(i32 { mag: 0, sign: false });
        data.append(i32 { mag: 1, sign: false });
        data.append(i32 { mag: 1, sign: true });

        let extra = Option::<ExtraParams>::None(());

        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

        return tensor;
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_acos_test() {
        let tensor = i32_tensor_1x3_helper_in_test();
        let result = tensor.acos().data;

        assert((*result.at(0).mag).into() == 102943, 'result[0] = 1.5707...');
        assert((*result.at(1).mag).into() == 0, 'result[1] = 0');
        assert((*result.at(2).mag).into() == 205887, 'result[2] = 3.141...');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_acos_fail() {
        let tensor = i32_tensor_1x3_helper();
        let result = tensor.acos().data;
    }
}
// // ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    use orion::tests::helpers::tensor::i32::i32_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};

    fn i32_tensor_2x2_helper_in_test() -> Tensor<i32> {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(i32 { mag: 0, sign: false });
        data.append(i32 { mag: 1, sign: false });
        data.append(i32 { mag: 1, sign: true });
        data.append(i32 { mag: 0, sign: false });

        let extra = Option::<ExtraParams>::None(());

        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

        return tensor;
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_acos_test() {
        let tensor = i32_tensor_2x2_helper_in_test();
        let result = tensor.acos().data;

        assert((*result.at(0).mag).into() == 102943, 'result[0] = 1.5707...');
        assert((*result.at(1).mag).into() == 0, 'result[1] = 0');
        assert((*result.at(2).mag).into() == 205887, 'result[2] = 3.141...');
        assert((*result.at(3).mag).into() == 102943, 'result[3] = 1.5707...');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_acos_fail() {
        let tensor = i32_tensor_2x2_helper();
        let result = tensor.acos().data;
    }
}
// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::i32::i32_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_acos_fail() {
        let tensor = i32_tensor_2x2x2_helper();
        let result = tensor.acos().data;
    }
}

