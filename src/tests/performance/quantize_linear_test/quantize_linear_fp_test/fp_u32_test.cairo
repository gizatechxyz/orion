#[cfg(test)]
mod fp8x23 {
    use array::ArrayTrait;
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;

    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
    use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;
    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams, Tensor};
    use orion::performance::core::PerfomanceTrait;
    use orion::performance::implementations::impl_performance_fp::Performance_fp_u32;

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
        data.append(FixedTrait::new_unscaled(254, false));
        data.append(FixedTrait::new_unscaled(1000, false));
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
        data.append(FixedTrait::new_unscaled(128, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
        let y_zero_point = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        let y: Tensor<u32> = x.quantize_linear(@y_scale, @y_zero_point);

        assert((*y.data[0]) == 128, '*result[0] == 128');
        assert((*y.data[1]) == 129, '*result[1] == 129');
        assert((*y.data[2]) == 129, '*result[2] == 129');
        assert((*y.data[3]) == 255, '*result[3] == 255');
        assert((*y.data[4]) == 255, '*result[4] == 255');
        assert((*y.data[5]) == 255, '*result[5] == 255');
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
        data.append(FixedTrait::new_unscaled(162, false));
        data.append(FixedTrait::new_unscaled(10, false));
        data.append(FixedTrait::new_unscaled(100, false));
        data.append(FixedTrait::new_unscaled(232, false));
        data.append(FixedTrait::new_unscaled(20, false));
        data.append(FixedTrait::new_unscaled(50, false));
        data.append(FixedTrait::new_unscaled(76, false));
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(252, false));
        data.append(FixedTrait::new_unscaled(32, false));
        data.append(FixedTrait::new_unscaled(44, false));
        data.append(FixedTrait::new_unscaled(245, false));
        data.append(FixedTrait::new_unscaled(485, false));
        data.append(FixedTrait::new_unscaled(960, false));
        data.append(FixedTrait::new_unscaled(270, false));
        data.append(FixedTrait::new_unscaled(375, false));
        data.append(FixedTrait::new_unscaled(470, false));
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

        let y: Tensor<u32> = x.quantize_linear(@y_scale, @y_zero_point);

        assert((*y.data[0]).into() == 165, '*result[0] == 165');
        assert((*y.data[1]).into() == 89, '*result[1] == 89');
        assert((*y.data[2]).into() == 134, '*result[2] == 134');
        assert((*y.data[3]).into() == 200, '*result[3] == 200');
        assert((*y.data[4]).into() == 94, '*result[4] == 94');
        assert((*y.data[5]).into() == 109, '*result[5] == 109');
        assert((*y.data[6]).into() == 43, '*result[6] == 43');
        assert((*y.data[7]).into() == 24, '*result[7] == 24');
        assert((*y.data[8]).into() == 24, '*result[8] == 24');
        assert((*y.data[9]).into() == 87, '*result[9] == 87');
        assert((*y.data[10]).into() == 32, '*result[10] == 32');
        assert((*y.data[11]).into() == 35, '*result[11] == 35');
        assert((*y.data[12]).into() == 245, '*result[12] == 245');
        assert((*y.data[13]).into() == 255, '*result[13] == 255');
        assert((*y.data[14]).into() == 255, '*result[14] == 255');
        assert((*y.data[15]).into() == 250, '*result[15] == 250');
        assert((*y.data[16]).into() == 255, '*result[16] == 255');
        assert((*y.data[17]).into() == 255, '*result[17] == 255');
    }
}


#[cfg(test)]
mod fp16x16 {
    use array::ArrayTrait;
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;

    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;
    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams, Tensor};
    use orion::performance::core::PerfomanceTrait;
    use orion::performance::implementations::impl_performance_fp::Performance_fp_u32;

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
        data.append(FixedTrait::new_unscaled(254, false));
        data.append(FixedTrait::new_unscaled(1000, false));
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
        data.append(FixedTrait::new_unscaled(128, false));
        let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
        let y_zero_point = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));

        let y: Tensor<u32> = x.quantize_linear(@y_scale, @y_zero_point);

        assert((*y.data[0]) == 128, '*result[0] == 128');
        assert((*y.data[1]) == 129, '*result[1] == 129');
        assert((*y.data[2]) == 129, '*result[2] == 129');
        assert((*y.data[3]) == 255, '*result[3] == 255');
        assert((*y.data[4]) == 255, '*result[4] == 255');
        assert((*y.data[5]) == 255, '*result[5] == 255');
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
        data.append(FixedTrait::new_unscaled(162, false));
        data.append(FixedTrait::new_unscaled(10, false));
        data.append(FixedTrait::new_unscaled(100, false));
        data.append(FixedTrait::new_unscaled(232, false));
        data.append(FixedTrait::new_unscaled(20, false));
        data.append(FixedTrait::new_unscaled(50, false));
        data.append(FixedTrait::new_unscaled(76, false));
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(252, false));
        data.append(FixedTrait::new_unscaled(32, false));
        data.append(FixedTrait::new_unscaled(44, false));
        data.append(FixedTrait::new_unscaled(245, false));
        data.append(FixedTrait::new_unscaled(485, false));
        data.append(FixedTrait::new_unscaled(960, false));
        data.append(FixedTrait::new_unscaled(270, false));
        data.append(FixedTrait::new_unscaled(375, false));
        data.append(FixedTrait::new_unscaled(470, false));
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

        let y: Tensor<u32> = x.quantize_linear(@y_scale, @y_zero_point);

        assert((*y.data[0]).into() == 165, '*result[0] == 165');
        assert((*y.data[1]).into() == 89, '*result[1] == 89');
        assert((*y.data[2]).into() == 134, '*result[2] == 134');
        assert((*y.data[3]).into() == 200, '*result[3] == 200');
        assert((*y.data[4]).into() == 94, '*result[4] == 94');
        assert((*y.data[5]).into() == 109, '*result[5] == 109');
        assert((*y.data[6]).into() == 43, '*result[6] == 43');
        assert((*y.data[7]).into() == 24, '*result[7] == 24');
        assert((*y.data[8]).into() == 24, '*result[8] == 24');
        assert((*y.data[9]).into() == 87, '*result[9] == 87');
        assert((*y.data[10]).into() == 32, '*result[10] == 32');
        assert((*y.data[11]).into() == 35, '*result[11] == 35');
        assert((*y.data[12]).into() == 245, '*result[12] == 245');
        assert((*y.data[13]).into() == 255, '*result[13] == 255');
        assert((*y.data[14]).into() == 255, '*result[14] == 255');
        assert((*y.data[15]).into() == 250, '*result[15] == 250');
        assert((*y.data[16]).into() == 255, '*result[16] == 255');
        assert((*y.data[17]).into() == 255, '*result[17] == 255');
    }
}

