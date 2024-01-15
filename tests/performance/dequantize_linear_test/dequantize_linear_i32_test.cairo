use core::debug::{PrintTrait};
use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::traits::Into;

use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::I8Tensor;
use orion::operators::tensor::I32Tensor;
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000)]
fn dequantize_linear() {
    // X
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    let mut data = ArrayTrait::<i8>::new();
    data.append(0);
    data.append(3);
    data.append(125);
    data.append(127);

    let x = TensorTrait::new(shape.span(), data.span());

    // XSCALE
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(2);

    let x_scale = TensorTrait::new(shape.span(), data.span());

    // ZEROPOINT
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(0);

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
    data.append(3);
    data.append(89);
    data.append(34);
    data.append(127);
    data.append(74);
    data.append(59);
    data.append(5);
    data.append(24);
    data.append(24);
    data.append(87);
    data.append(32);
    data.append(13);
    data.append(127);
    data.append(99);
    data.append(4);
    data.append(127);
    data.append(121);
    data.append(102);

    let x = TensorTrait::new(shape.span(), data.span());

    // XSCALE
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(1);
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(2);
    data.append(4);
    data.append(5);

    let x_scale = TensorTrait::new(shape.span(), data.span());

    // ZEROPOINT
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(1);
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(1);
    data.append(2);
    data.append(3);

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
