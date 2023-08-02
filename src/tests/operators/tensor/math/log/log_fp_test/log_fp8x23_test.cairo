// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::numbers::fixed_point::{
        core::{FixedTrait, FixedType, FixedImpl}, implementations::impl_8x23::FP8x23Impl
    };
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

    use debug::PrintTrait;
    #[test]
    #[available_gas(20000000)]
    fn tensor_log_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(4);

        let mut data = ArrayTrait::new();
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(100, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

        let result = tensor.log().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 5814556, 'result[1] = 0.69315');
        assert((*result.at(2).mag).into() == 9215831, 'result[2] = 1.0986');
        assert((*result.at(3).mag).into() == 38630976, 'result[3] = 4.60517');
    }
}

// ===== 2D ===== //

mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::numbers::fixed_point::{
        core::{FixedTrait, FixedType, FixedImpl}, implementations::impl_8x23::FP8x23Impl
    };
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

    #[test]
    #[available_gas(20000000)]
    fn tensor_log_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(100, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

        let result = tensor.log().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 5814556, 'result[1] = 0.69315');
        assert((*result.at(2).mag).into() == 9215831, 'result[2] = 1.0986');
        assert((*result.at(3).mag).into() == 38630976, 'result[3] = 4.60517');
    }
}
// // ===== 3D ===== //

mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::numbers::fixed_point::{
        core::{FixedTrait, FixedType, FixedImpl}, implementations::impl_8x23::FP8x23Impl
    };
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

    #[test]
    #[available_gas(20000000)]
    fn tensor_log_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(100, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(100, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(100, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

        let result = tensor.log().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 5814556, 'result[1] = 0.69315');
        assert((*result.at(2).mag).into() == 9215831, 'result[2] = 1.0986');
        assert((*result.at(3).mag).into() == 38630976, 'result[3] = 4.60517');
        assert((*result.at(4).mag).into() == 9215831, 'result[4] = 1.0986');
        assert((*result.at(5).mag).into() == 38630976, 'result[5] = 4.60517');
        assert((*result.at(6).mag).into() == 9215831, 'result[6] = 1.0986');
        assert((*result.at(7).mag).into() == 38630976, 'result[7] = 4.60517');
    }
}

