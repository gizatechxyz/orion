use core::debug::{PrintTrait};
use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32, i8::i8};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_i8_fp16x16::Tensor_i8_fp16x16;
use orion::operators::tensor::implementations::tensor_i32_fp16x16::Tensor_i32_fp16x16;
use orion::operators::tensor::core::{TensorTrait, Tensor};

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
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(2, false));
    
    let x_scale = TensorTrait::new(shape.span(), data.span());

    // ZEROPOINT
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(0, false));
    
    let x_zero_point = TensorTrait::new(shape.span(), data.span());

    let y: Tensor<i32> = x.dequantize_linear(@x_scale, @x_zero_point);

    assert((*y.data[0]).into() == 0, '*result[0] == 0');
    assert((*y.data[1]).into() == 6, '*result[1] == 6');
    assert((*y.data[2]).into() == 250, '*result[2] == 250');
    assert((*y.data[3]).into() == 254, '*result[3] == 254');
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
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(2, false));
    data.append(IntegerTrait::new(4, false));
    data.append(IntegerTrait::new(5, false));
    
    let x_scale = TensorTrait::new(shape.span(), data.span());

    // ZEROPOINT
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(1);
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(1, false));
    data.append(IntegerTrait::new(2, false));
    data.append(IntegerTrait::new(3, false));
    
    let x_zero_point = TensorTrait::new(shape.span(), data.span());

    let y: Tensor<i32> = x.dequantize_linear(@x_scale, @x_zero_point);

    assert((*y.data[0]).into() == 4, '*result[0] == 162');
    assert((*y.data[1]).into() == 176, '*result[1] == 10');
    assert((*y.data[2]).into() == 66, '*result[2] == 100');
    assert((*y.data[3]).into() == 252, '*result[3] == 232');
    assert((*y.data[4]).into() == 146, '*result[4] == 20');
    assert((*y.data[5]).into() == 116, '*result[5] == 50');
    assert((*y.data[6]).into() == 12, '*result[6] == 76');
    assert((*y.data[7]).into() == 88, '*result[7] == 0');
    assert((*y.data[8]).into() == 88, '*result[8] == 0');
    assert((*y.data[9]).into() == 340, '*result[9] == 252');
    assert((*y.data[10]).into() == 120, '*result[10] == 32');
    assert((*y.data[11]).into() == 44, '*result[11] == 44');
    assert((*y.data[12]).into() == 620, '*result[12] == 245');
    assert((*y.data[13]).into() == 480, '*result[13] == 485');
    assert((*y.data[14]).into() == 5, '*result[14] == 960');
    assert((*y.data[15]).into() == 620, '*result[15] == 270');
    assert((*y.data[16]).into() == 590, '*result[16] == 375');
    assert((*y.data[17]).into() == 495, '*result[17] == 470');
}
