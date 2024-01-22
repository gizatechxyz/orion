#[cfg(test)]
mod i32 {
    use core::array::ArrayTrait;
    use core::array::SpanTrait;
    use core::traits::Into;
    use core::debug::PrintTrait;

    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32, i8::i8};
    use orion::numbers::fixed_point::core::{FixedTrait};
    use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};

    #[test]
    #[available_gas(2000000)]
    fn dynamic_quantize_linear() {
        // X
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(6);
        let mut data = ArrayTrait::<i32>::new();
        data.append(IntegerTrait::new(1, false));  // 1
        data.append(IntegerTrait::new(66, false));  // 6
        data.append(IntegerTrait::new(966, false));  // 9
        data.append(IntegerTrait::new(270, false)); // 27
        data.append(IntegerTrait::new(911, true));  // -11
        data.append(IntegerTrait::new(661, true));  // -66

        let x = TensorTrait::new(shape.span(), data.span());

        let (y, y_scale, y_zero_point) = x.dynamic_quantize_linear();

        assert((*(y_scale.data).at(0)).mag == 7, '*y_scale[0].mag == 7');
        assert((*(y_scale.data).at(0)).sign == false, '*y_scale[0].sign == false');
        assert((*(y_zero_point.data).at(0)).mag == 130, '*y_zero_point[0].mag == 130');
        assert((*(y_zero_point.data).at(0)).sign == false, '*y_zero_point[0].sign == false');
        assert((*(y.data).at(0)).mag == 130, '*result[0] == 130');
        assert((*(y.data).at(1)).mag == 139, '*result[1] == 139');
        assert((*(y.data).at(2)).mag == 255, '*result[2] == 255');
        assert((*(y.data).at(3)).mag == 168, '*result[3] == 168');
        assert((*(y.data).at(4)).mag == 0, '*result[4] == 0');
        assert((*(y.data).at(5)).mag == 36, '*result[5] == 36');
    }
}
