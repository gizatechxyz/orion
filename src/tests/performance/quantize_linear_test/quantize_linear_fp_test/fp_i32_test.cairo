#[cfg(test)]
mod fp8x23 {
    use array::ArrayTrait;
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;

    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32, i8::i8};
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
    use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;
    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams, Tensor};
    use orion::performance::core::PerfomanceTrait;
    use orion::performance::implementations::impl_performance_fp::Performance_fp;

    #[test]
    #[available_gas(2000000)]
    fn quantize_linear() {
        // X
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(6);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(1000, false));
        data.append(FixedTrait::new_unscaled(254, true));
        data.append(FixedTrait::new_unscaled(1000, true));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let x = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        // YSCALE
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(2, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let y_scale = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        // ZEROPOINT
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(1, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let y_zero_point = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        let y: Tensor<i8> = x.quantize_linear(@y_scale, @y_zero_point);

        assert((*y.data[0]).into() == 1, '*result[0] == 1');
        assert((*y.data[1]).into() == 2, '*result[1] == 2');
        assert((*y.data[2]).into() == 2, '*result[2] == 2');
        assert((*y.data[3]).into() == 127, '*result[3] == 127');
        assert((*y.data[4]).into() == -126, '*result[4] == -126');
        assert((*y.data[5]).into() == -128, '*result[5] == -128');
    }

    #[test]
    #[available_gas(20000000)]
    fn per_axis() {
        // X
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        shape.append(3);
        shape.append(3);
        shape.append(2);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(162, true));
        data.append(FixedTrait::new_unscaled(10, false));
        data.append(FixedTrait::new_unscaled(100, true));
        data.append(FixedTrait::new_unscaled(232, false));
        data.append(FixedTrait::new_unscaled(20, true));
        data.append(FixedTrait::new_unscaled(50, true));
        data.append(FixedTrait::new_unscaled(76, true));
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(252, false));
        data.append(FixedTrait::new_unscaled(32, false));
        data.append(FixedTrait::new_unscaled(44, true));
        data.append(FixedTrait::new_unscaled(245, false));
        data.append(FixedTrait::new_unscaled(485, true));
        data.append(FixedTrait::new_unscaled(960, true));
        data.append(FixedTrait::new_unscaled(270, true));
        data.append(FixedTrait::new_unscaled(375, true));
        data.append(FixedTrait::new_unscaled(470, true));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let x = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        // YSCALE
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        shape.append(3);
        shape.append(1);
        shape.append(1);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(4, false));
        data.append(FixedTrait::new_unscaled(5, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let y_scale = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        // ZEROPOINT
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        shape.append(3);
        shape.append(1);
        shape.append(1);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(84, false));
        data.append(FixedTrait::new_unscaled(24, false));
        data.append(FixedTrait::new_unscaled(196, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let y_zero_point = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        let y: Tensor<i8> = x.quantize_linear(@y_scale, @y_zero_point);

        assert((*y.data[0]).into() == 3, '*result[0] == 3');
        assert((*y.data[1]).into() == 89, '*result[1] == 89');
        assert((*y.data[2]).into() == 34, '*result[2] == 34');
        assert((*y.data[3]).into() == 127, '*result[3] == 127');
        assert((*y.data[4]).into() == 74, '*result[4] == 74');
        assert((*y.data[5]).into() == 59, '*result[5] == 59');
        assert((*y.data[6]).into() == 5, '*result[6] == 5');
        assert((*y.data[7]).into() == 24, '*result[7] == 24');
        assert((*y.data[8]).into() == 24, '*result[8] == 24');
        assert((*y.data[9]).into() == 87, '*result[9] == 87');
        assert((*y.data[10]).into() == 32, '*result[10] == 32');
        assert((*y.data[11]).into() == 13, '*result[11] == 13');
        assert((*y.data[12]).into() == 127, '*result[12] == 127');
        assert((*y.data[13]).into() == 99, '*result[13] == 99');
        assert((*y.data[14]).into() == 4, '*result[14] == 4');
        assert((*y.data[15]).into() == 127, '*result[15] == 127');
        assert((*y.data[16]).into() == 121, '*result[16] == 121');
        assert((*y.data[17]).into() == 102, '*result[17] == 102');
    }
}


#[cfg(test)]
mod fp16x16 {
    use array::ArrayTrait;
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;

    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32, i8::i8};
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;
    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams, Tensor};
    use orion::performance::core::PerfomanceTrait;
    use orion::performance::implementations::impl_performance_fp::Performance_fp;

    #[test]
    #[available_gas(2000000)]
    fn quantize_linear() {
        // X
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(6);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(1000, false));
        data.append(FixedTrait::new_unscaled(254, true));
        data.append(FixedTrait::new_unscaled(1000, true));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
        let x = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        // YSCALE
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(2, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
        let y_scale = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        // ZEROPOINT
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(1, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
        let y_zero_point = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        let y: Tensor<i8> = x.quantize_linear(@y_scale, @y_zero_point);

        assert((*y.data[0]).into() == 1, '*result[0] == 1');
        assert((*y.data[1]).into() == 2, '*result[1] == 2');
        assert((*y.data[2]).into() == 2, '*result[2] == 2');
        assert((*y.data[3]).into() == 127, '*result[3] == 127');
        assert((*y.data[4]).into() == -126, '*result[4] == -126');
        assert((*y.data[5]).into() == -128, '*result[5] == -128');
    }

    #[test]
    #[available_gas(20000000)]
    fn per_axis() {
        // X
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        shape.append(3);
        shape.append(3);
        shape.append(2);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(162, true));
        data.append(FixedTrait::new_unscaled(10, false));
        data.append(FixedTrait::new_unscaled(100, true));
        data.append(FixedTrait::new_unscaled(232, false));
        data.append(FixedTrait::new_unscaled(20, true));
        data.append(FixedTrait::new_unscaled(50, true));
        data.append(FixedTrait::new_unscaled(76, true));
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(252, false));
        data.append(FixedTrait::new_unscaled(32, false));
        data.append(FixedTrait::new_unscaled(44, true));
        data.append(FixedTrait::new_unscaled(245, false));
        data.append(FixedTrait::new_unscaled(485, true));
        data.append(FixedTrait::new_unscaled(960, true));
        data.append(FixedTrait::new_unscaled(270, true));
        data.append(FixedTrait::new_unscaled(375, true));
        data.append(FixedTrait::new_unscaled(470, true));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
        let x = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        // YSCALE
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        shape.append(3);
        shape.append(1);
        shape.append(1);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(4, false));
        data.append(FixedTrait::new_unscaled(5, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
        let y_scale = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        // ZEROPOINT
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        shape.append(3);
        shape.append(1);
        shape.append(1);
        let mut data = ArrayTrait::<FixedType>::new();
        data.append(FixedTrait::new_unscaled(84, false));
        data.append(FixedTrait::new_unscaled(24, false));
        data.append(FixedTrait::new_unscaled(196, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
        let y_zero_point = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        let y: Tensor<i8> = x.quantize_linear(@y_scale, @y_zero_point);

        assert((*y.data[0]).into() == 3, '*result[0] == 3');
        assert((*y.data[1]).into() == 89, '*result[1] == 89');
        assert((*y.data[2]).into() == 34, '*result[2] == 34');
        assert((*y.data[3]).into() == 127, '*result[3] == 127');
        assert((*y.data[4]).into() == 74, '*result[4] == 74');
        assert((*y.data[5]).into() == 59, '*result[5] == 59');
        assert((*y.data[6]).into() == 5, '*result[6] == 5');
        assert((*y.data[7]).into() == 24, '*result[7] == 24');
        assert((*y.data[8]).into() == 24, '*result[8] == 24');
        assert((*y.data[9]).into() == 87, '*result[9] == 87');
        assert((*y.data[10]).into() == 32, '*result[10] == 32');
        assert((*y.data[11]).into() == 13, '*result[11] == 13');
        assert((*y.data[12]).into() == 127, '*result[12] == 127');
        assert((*y.data[13]).into() == 99, '*result[13] == 99');
        assert((*y.data[14]).into() == 4, '*result[14] == 4');
        assert((*y.data[15]).into() == 127, '*result[15] == 127');
        assert((*y.data[16]).into() == 121, '*result[16] == 121');
        assert((*y.data[17]).into() == 102, '*result[17] == 102');
    }
}
