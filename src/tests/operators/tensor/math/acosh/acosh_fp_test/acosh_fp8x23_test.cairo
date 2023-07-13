// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_1x3_helper;
    use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};

    #[test]
    #[available_gas(20000000)]
    fn tensor_acosh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);

        let mut data = ArrayTrait::new();
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));
        let result = tensor.acosh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 11047444, 'result[1] = 1.31696...');
        assert((*result.at(2).mag).into() == 14787433, 'result[2] = 1.76275...');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};

    #[test]
    #[available_gas(20000000)]
    fn tensor_acosh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(4, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));
        let result = tensor.acosh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 11047444, 'result[1] = 1.31696...');
        assert((*result.at(2).mag).into() == 14787433, 'result[2] = 1.76275...');
        assert((*result.at(3).mag).into() == 17309707, 'result[3] = 2.06344...');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;
    use array::ArrayTrait;


    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};

    #[test]
    #[available_gas(20000000)]
    fn tensor_acosh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(4, false));
        data.append(FixedTrait::new_unscaled(5, false));
        data.append(FixedTrait::new_unscaled(6, false));
        data.append(FixedTrait::new_unscaled(7, false));
        data.append(FixedTrait::new_unscaled(8, false));

        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));
        let result = tensor.acosh().data;

        let a0 = *result.at(0).mag.into();
        let a1 = *result.at(1).mag.into();
        let a2 = *result.at(2).mag.into();
        let a3 = *result.at(3).mag.into();
        let a4 = *result.at(4).mag.into();
        let a5 = *result.at(5).mag.into();
        let a6 = *result.at(6).mag.into();
        let a7 = *result.at(7).mag.into();

        assert((*result.at(0).mag).into() == 0, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 11047444, 'result[1] = 1.31696...');
        assert((*result.at(2).mag).into() == 14787433, 'result[2] = 1.76275...');
        assert((*result.at(3).mag).into() == 17309707, 'result[3] = 2.06344...');
        assert((*result.at(4).mag).into() == 19230472, 'result[4] = 2.29243...');
        assert((*result.at(5).mag).into() == 20786291, 'result[5] = 2.47789...');
        assert((*result.at(6).mag).into() == 22095298, 'result[6] = 2.63392..');
        assert((*result.at(7).mag).into() == 23225568, 'result[7] = 2.76866...');
    }
}

