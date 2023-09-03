#[cfg(test)]
mod fp8x23 {
    use array::ArrayTrait;
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;

    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
    use orion::numbers::fixed_point::core::{FixedTrait};
    use orion::numbers::fixed_point::implementations::fp8x23::core::{FP8x23Impl, FP8x23PartialEq};
    use orion::operators::tensor::I8Tensor;
    use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;
    use orion::operators::tensor::{TensorTrait, Tensor};
    use orion::numbers::FP8x23;

    #[test]
    #[available_gas(2000000)]
    fn dequantize_linear() {
        // X
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(4);
        let mut data = ArrayTrait::<i8>::new();
        data.append(IntegerTrait::new(0, false));
        data.append(IntegerTrait::new(3, false));
        data.append(IntegerTrait::new(125, false));
        data.append(IntegerTrait::new(127, false));
        
        let x = TensorTrait::new(shape.span(), data.span());

        // XSCALE
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        let mut data = ArrayTrait::<FP8x23>::new();
        data.append(FixedTrait::new_unscaled(2, false));
        
        let x_scale = TensorTrait::new(shape.span(), data.span());

        // ZEROPOINT
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        let mut data = ArrayTrait::<FP8x23>::new();
        data.append(FixedTrait::new_unscaled(0, false));
        
        let x_zero_point = TensorTrait::new(shape.span(), data.span());

        let y: Tensor<FP8x23> = x.dequantize_linear(@x_scale, @x_zero_point);

        assert(
            (*y.data[0]).into() == FixedTrait::<FP8x23, u32>::new_unscaled(0, false),
            '*result[0] == 0'
        );
        assert(
            (*y.data[1]).into() == FixedTrait::<FP8x23, u32>::new_unscaled(6, false),
            '*result[1] == 6'
        );
        assert(
            (*y.data[2]).into() == FixedTrait::<FP8x23, u32>::new_unscaled(250, false),
            '*result[2] == 250'
        );
        assert(
            (*y.data[3]).into() == FixedTrait::<FP8x23, u32>::new_unscaled(254, false),
            '*result[3] == 254'
        );
    }
// TODO: fix it when move performance tests in nodes.
// #[test]
// #[available_gas(20000000)]
// fn per_axis() {
//     // X
//     let mut shape = ArrayTrait::<usize>::new();
//     shape.append(1);
//     shape.append(3);
//     shape.append(3);
//     shape.append(2);
//     let mut data = ArrayTrait::<i8>::new();
//     data.append(IntegerTrait::new(3, false));
//     data.append(IntegerTrait::new(89, false));
//     data.append(IntegerTrait::new(34, false));
//     data.append(IntegerTrait::new(127, false));
//     data.append(IntegerTrait::new(74, false));
//     data.append(IntegerTrait::new(59, false));
//     data.append(IntegerTrait::new(5, false));
//     data.append(IntegerTrait::new(24, false));
//     data.append(IntegerTrait::new(24, false));
//     data.append(IntegerTrait::new(87, false));
//     data.append(IntegerTrait::new(32, false));
//     data.append(IntegerTrait::new(13, false));
//     data.append(IntegerTrait::new(127, false));
//     data.append(IntegerTrait::new(99, false));
//     data.append(IntegerTrait::new(4, false));
//     data.append(IntegerTrait::new(127, false));
//     data.append(IntegerTrait::new(121, false));
//     data.append(IntegerTrait::new(102, false));
//     
//     let x = TensorTrait::new(shape.span(), data.span());

//     // XSCALE
//     let mut shape = ArrayTrait::<usize>::new();
//     shape.append(1);
//     shape.append(3);
//     shape.append(1);
//     shape.append(1);
//     let mut data = ArrayTrait::<FixedType>::new();
//     data.append(FixedTrait::new_unscaled(2, false));
//     data.append(FixedTrait::new_unscaled(4, false));
//     data.append(FixedTrait::new_unscaled(5, false));
//     
//     let x_scale = TensorTrait::new(shape.span(), data.span());

//     // ZEROPOINT
//     let mut shape = ArrayTrait::<usize>::new();
//     shape.append(1);
//     shape.append(3);
//     shape.append(1);
//     shape.append(1);
//     let mut data = ArrayTrait::<FixedType>::new();
//     data.append(FixedTrait::new_unscaled(1, false));
//     data.append(FixedTrait::new_unscaled(2, false));
//     data.append(FixedTrait::new_unscaled(3, false));
//     
//     let x_zero_point = TensorTrait::new(shape.span(), data.span());

//     let y: Tensor<FixedType> = x.dequantize_linear(@x_scale, @x_zero_point);

//     assert((*y.data[0]).into() == FixedTrait::new_unscaled(4, false), '*result[0] == 162');
//     assert((*y.data[1]).into() == FixedTrait::new_unscaled(176, false), '*result[1] == 10');
//     assert((*y.data[2]).into() == FixedTrait::new_unscaled(66, false), '*result[2] == 100');
//     assert((*y.data[3]).into() == FixedTrait::new_unscaled(252, false), '*result[3] == 232');
//     assert((*y.data[4]).into() == FixedTrait::new_unscaled(146, false), '*result[4] == 20');
//     assert((*y.data[5]).into() == FixedTrait::new_unscaled(116, false), '*result[5] == 50');
//     assert((*y.data[6]).into() == FixedTrait::new_unscaled(12, false), '*result[6] == 76');
//     assert((*y.data[7]).into() == FixedTrait::new_unscaled(88, false), '*result[7] == 0');
//     assert((*y.data[8]).into() == FixedTrait::new_unscaled(88, false), '*result[8] == 0');
//     assert((*y.data[9]).into() == FixedTrait::new_unscaled(340, false), '*result[9] == 252');
//     assert((*y.data[10]).into() == FixedTrait::new_unscaled(120, false), '*result[10] == 32');
//     assert((*y.data[11]).into() == FixedTrait::new_unscaled(44, false), '*result[11] == 44');
//     assert((*y.data[12]).into() == FixedTrait::new_unscaled(620, false), '*result[12] == 245');
//     assert((*y.data[13]).into() == FixedTrait::new_unscaled(480, false), '*result[13] == 485');
//     assert((*y.data[14]).into() == FixedTrait::new_unscaled(5, false), '*result[14] == 960');
//     assert((*y.data[15]).into() == FixedTrait::new_unscaled(620, false), '*result[15] == 270');
//     assert((*y.data[16]).into() == FixedTrait::new_unscaled(590, false), '*result[16] == 375');
//     assert((*y.data[17]).into() == FixedTrait::new_unscaled(495, false), '*result[17] == 470');
// }
}

#[cfg(test)]
mod fp16x16 {
    use array::ArrayTrait;
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;


    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
    use orion::numbers::fixed_point::core::{FixedTrait};
    use orion::numbers::fixed_point::implementations::fp16x16::core::{
        FP16x16Impl, FP16x16PartialEq
    };
    use orion::operators::tensor::I8Tensor;
    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::{TensorTrait, Tensor};
    use orion::numbers::FP16x16;

    #[test]
    #[available_gas(2000000)]
    fn dequantize_linear() {
        // X
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(4);
        let mut data = ArrayTrait::<i8>::new();
        data.append(IntegerTrait::new(0, false));
        data.append(IntegerTrait::new(3, false));
        data.append(IntegerTrait::new(125, false));
        data.append(IntegerTrait::new(127, false));
        
        let x = TensorTrait::new(shape.span(), data.span());

        // XSCALE
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        let mut data = ArrayTrait::<FP16x16>::new();
        data.append(FixedTrait::new_unscaled(2, false));
        
        let x_scale = TensorTrait::new(shape.span(), data.span());

        // ZEROPOINT
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        let mut data = ArrayTrait::<FP16x16>::new();
        data.append(FixedTrait::new_unscaled(0, false));
        
        let x_zero_point = TensorTrait::new(shape.span(), data.span());

        let y: Tensor<FP16x16> = x.dequantize_linear(@x_scale, @x_zero_point);

        assert(
            (*y.data[0]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(0, false),
            '*result[0] == 0'
        );
        assert(
            (*y.data[1]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(6, false),
            '*result[1] == 6'
        );
        assert(
            (*y.data[2]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(250, false),
            '*result[2] == 250'
        );
        assert(
            (*y.data[3]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(254, false),
            '*result[3] == 254'
        );
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
        let mut data = ArrayTrait::<i8>::new();
        data.append(IntegerTrait::new(3, false));
        data.append(IntegerTrait::new(89, false));
        data.append(IntegerTrait::new(34, false));
        data.append(IntegerTrait::new(127, false));
        data.append(IntegerTrait::new(74, false));
        data.append(IntegerTrait::new(59, false));
        data.append(IntegerTrait::new(5, false));
        data.append(IntegerTrait::new(24, false));
        data.append(IntegerTrait::new(24, false));
        data.append(IntegerTrait::new(87, false));
        data.append(IntegerTrait::new(32, false));
        data.append(IntegerTrait::new(13, false));
        data.append(IntegerTrait::new(127, false));
        data.append(IntegerTrait::new(99, false));
        data.append(IntegerTrait::new(4, false));
        data.append(IntegerTrait::new(127, false));
        data.append(IntegerTrait::new(121, false));
        data.append(IntegerTrait::new(102, false));
        
        let x = TensorTrait::new(shape.span(), data.span());

        // XSCALE
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        shape.append(3);
        shape.append(1);
        shape.append(1);
        let mut data = ArrayTrait::<FP16x16>::new();
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(4, false));
        data.append(FixedTrait::new_unscaled(5, false));
        
        let x_scale = TensorTrait::new(shape.span(), data.span());

        // ZEROPOINT
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(1);
        shape.append(3);
        shape.append(1);
        shape.append(1);
        let mut data = ArrayTrait::<FP16x16>::new();
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        
        let x_zero_point = TensorTrait::new(shape.span(), data.span());

        let y: Tensor<FP16x16> = x.dequantize_linear(@x_scale, @x_zero_point);

        assert(
            (*y.data[0]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(4, false),
            '*result[0] == 162'
        );
        assert(
            (*y.data[1]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(176, false),
            '*result[1] == 10'
        );
        assert(
            (*y.data[2]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(66, false),
            '*result[2] == 100'
        );
        assert(
            (*y.data[3]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(252, false),
            '*result[3] == 232'
        );
        assert(
            (*y.data[4]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(146, false),
            '*result[4] == 20'
        );
        assert(
            (*y.data[5]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(116, false),
            '*result[5] == 50'
        );
        assert(
            (*y.data[6]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(12, false),
            '*result[6] == 76'
        );
        assert(
            (*y.data[7]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(88, false),
            '*result[7] == 0'
        );
        assert(
            (*y.data[8]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(88, false),
            '*result[8] == 0'
        );
        assert(
            (*y.data[9]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(340, false),
            '*result[9] == 252'
        );
        assert(
            (*y.data[10]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(120, false),
            '*result[10] == 32'
        );
        assert(
            (*y.data[11]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(44, false),
            '*result[11] == 44'
        );
        assert(
            (*y.data[12]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(620, false),
            '*result[12] == 245'
        );
        assert(
            (*y.data[13]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(480, false),
            '*result[13] == 485'
        );
        assert(
            (*y.data[14]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(5, false),
            '*result[14] == 960'
        );
        assert(
            (*y.data[15]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(620, false),
            '*result[15] == 270'
        );
        assert(
            (*y.data[16]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(590, false),
            '*result[16] == 375'
        );
        assert(
            (*y.data[17]).into() == FixedTrait::<FP16x16, u32>::new_unscaled(495, false),
            '*result[17] == 470'
        );
    }
}
