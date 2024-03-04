#[cfg(test)]
mod fp8x23 {
    use core::array::ArrayTrait;
    use core::array::SpanTrait;
    use core::traits::Into;
    use core::debug::PrintTrait;

    use orion::numbers::fixed_point::core::{FixedTrait};
    use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;
    use orion::operators::tensor::{TensorTrait, Tensor};
    use orion::numbers::FP8x23;

    #[test]
    #[available_gas(2000000)]
    fn dynamic_quantize_linear() {
        // X
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(6);
        let mut data = ArrayTrait::<FP8x23>::new();
        data.append(FixedTrait::new(0, false));
        data.append(FixedTrait::new(587203, false)); // 0.07
        data.append(FixedTrait::new(838861, false)); // 0.1
        data.append(FixedTrait::new(1677722, false)); // 0.2
        data.append(FixedTrait::new(4194304, false)); // 0.5
        data.append(FixedTrait::new(7549747, false)); // 0.9

        let x = TensorTrait::new(shape.span(), data.span());

        let (y, y_scale, y_zero_point) = x.dynamic_quantize_linear();

        assert((*(y_scale.data).at(0)).into() == 29606, '*y_scale[0].mag == 0.00353');
        assert((*(y_zero_point.data).at(0)).into() == 0, '*y_zero_point[0].mag == 0');
        assert((*(y.data).at(0)).into() == 0, '*result[0] == 0');
        assert((*(y.data).at(1)).into() == 19, '*result[1] == 19');
        assert((*(y.data).at(2)).into() == 28, '*result[2] == 28');
        assert((*(y.data).at(3)).into() == 56, '*result[3] == 56');
        assert((*(y.data).at(4)).into() == 141, '*result[4] == 141');
        assert((*(y.data).at(5)).into() == 255, '*result[5] == 255');
    }
}


#[cfg(test)]
mod fp16x16 {
    use core::array::ArrayTrait;
    use core::array::SpanTrait;
    use core::traits::Into;
    use core::debug::PrintTrait;

    use orion::numbers::fixed_point::core::{FixedTrait};
    use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;
    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::{TensorTrait, Tensor};
    use orion::numbers::FP16x16;

    #[test]
    #[available_gas(2000000)]
    fn dynamic_quantize_linear() {
        // X
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(6);
        let mut data = ArrayTrait::<FP16x16>::new();
        data.append(FixedTrait::new(10945, false)); // 0.167
        data.append(FixedTrait::new(190054, false)); // 2.9
        data.append(FixedTrait::new_unscaled(3, false)); // 3.0
        data.append(FixedTrait::new(229376, false)); // 3.5
        data.append(FixedTrait::new_unscaled(3, true)); // -3.0
        data.append(FixedTrait::new(229376, true)); // -3.5

        let x = TensorTrait::new(shape.span(), data.span());

        let (y, y_scale, y_zero_point) = x.dynamic_quantize_linear();

        assert((*(y_scale.data).at(0)).into() == 1799, '*y_scale[0].mag == 0.02745');
        assert((*(y_zero_point.data).at(0)).into() == 8355967, '*y_zero_point[0].mag == 128');
        assert((*(y.data).at(0)).into() == 133, '*result[0] == 134');
        assert((*(y.data).at(1)).into() == 233, '*result[1] == 233');
        assert((*(y.data).at(2)).into() == 236, '*result[2] == 237');
        assert((*(y.data).at(3)).into() == 255, '*result[3] == 255');
        assert((*(y.data).at(4)).into() == 18, '*result[4] == -18');
        assert((*(y.data).at(5)).into() == 0, '*result[5] == -0');
    }
}
