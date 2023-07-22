// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};

    fn fp_tensor_1x3_helper_in_test() -> Tensor<FixedType> {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);

        let mut data = ArrayTrait::new();
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(1, true));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

        return tensor;
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_acos_test() {
        let tensor = fp_tensor_1x3_helper_in_test();
        let result = tensor.acos().data;

        assert((*result.at(0).mag).into() == 102943, 'result[0] = 1.5707...');
        assert((*result.at(1).mag).into() == 0, 'result[1] = 0');
        assert((*result.at(2).mag).into() == 205887, 'result[2] = 3.141...');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_acos_fail() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.acos().data;
    }
}
// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};

    fn fp_tensor_2x2_helper_in_test() -> Tensor<FixedType> {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(1, true));
        data.append(FixedTrait::from_felt(32768)); //0.5

        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

        return tensor;
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_acos_test() {
        let tensor = fp_tensor_2x2_helper_in_test();
        let result = tensor.acos().data;

        assert((*result.at(0).mag).into() == 102943, 'result[0] = 1.5707...');
        assert((*result.at(1).mag).into() == 0, 'result[1] = 0');
        assert((*result.at(2).mag).into() == 205887, 'result[2] = 3.141...');
        assert((*result.at(3).mag).into() == 68317, 'result[2] = 1.047...');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_acos_fail() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.acos().data;
    }
}
// // ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};

    fn fp_tensor_2x2x2_helper_in_test() -> Tensor<FixedType> {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();

        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(1, true));
        data.append(FixedTrait::from_felt(32768)); //0.5
        data.append(FixedTrait::from_felt(16384)); //0.25
        data.append(FixedTrait::from_felt(6553)); //0.1
        data.append(FixedTrait::from_felt(-6553)); //-0.1
        data.append(FixedTrait::from_felt(-32768)); //-0.5

        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

        return tensor;
    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_acos_test() {
        let tensor = fp_tensor_2x2x2_helper_in_test();
        let result = tensor.acos().data;

        assert((*result.at(0).mag).into() == 102943, 'result[0] = 1.5707...');
        assert((*result.at(1).mag).into() == 0, 'result[1] = 0');
        assert((*result.at(2).mag).into() == 205887, 'result[2] = 3.141...');
        assert((*result.at(3).mag).into() == 68317, 'result[2] = 1.047...');
        assert((*result.at(4).mag).into() == 85584, 'result[4] = 1.318...');
        assert((*result.at(5).mag).into() == 94985, 'result[5] = 1.4706...');
        assert((*result.at(6).mag).into() == 110902, 'result[6] =  1.6709...');
        assert((*result.at(7).mag).into() == 137570, 'result[7] =  2.0943...');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_acos_fail() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.acos().data;
    }
}

