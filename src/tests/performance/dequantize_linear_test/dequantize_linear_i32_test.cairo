use core::debug::{PrintTrait};
use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::core::{TensorTrait, ExtraParams, Tensor};
use orion::performance::core::PerfomanceTrait;
use orion::performance::implementations::impl_performance_i32::Performance_i32;

#[test]
#[available_gas(2000000)]
fn dequantize_linear() {
    // X
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(0, false));
    data.append(IntegerTrait::new(3, false));
    data.append(IntegerTrait::new(128, false));
    data.append(IntegerTrait::new(255, false));
    let extra = Option::<ExtraParams>::None(());
    let x = TensorTrait::new(shape.span(), data.span(), extra);

    // XSCALE
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(2, false));
    let extra = Option::<ExtraParams>::None(());
    let x_scale = TensorTrait::new(shape.span(), data.span(), extra);

    // ZEROPOINT
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(128, false));
    let extra = Option::<ExtraParams>::None(());
    let x_zero_point = TensorTrait::new(shape.span(), data.span(), extra);

    let y: Tensor<i32> = x.dequantize_linear(@x_scale, @x_zero_point);

    assert((*y.data[0]).into() == -256, '*result[0] == -256');
    assert((*y.data[1]).into() == -250, '*result[1] == -250');
    assert((*y.data[2]).into() == 0, '*result[2] == 0');
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
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(3, false));
    data.append(IntegerTrait::new(89, false));
    data.append(IntegerTrait::new(34, false));
    data.append(IntegerTrait::new(200, false));
    data.append(IntegerTrait::new(74, false));
    data.append(IntegerTrait::new(59, false));
    data.append(IntegerTrait::new(5, false));
    data.append(IntegerTrait::new(24, false));
    data.append(IntegerTrait::new(24, false));
    data.append(IntegerTrait::new(87, false));
    data.append(IntegerTrait::new(32, false));
    data.append(IntegerTrait::new(13, false));
    data.append(IntegerTrait::new(245, false));
    data.append(IntegerTrait::new(99, false));
    data.append(IntegerTrait::new(4, false));
    data.append(IntegerTrait::new(142, false));
    data.append(IntegerTrait::new(121, false));
    data.append(IntegerTrait::new(102, false));
    let extra = Option::<ExtraParams>::None(());
    let x = TensorTrait::new(shape.span(), data.span(), extra);

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
    let extra = Option::<ExtraParams>::None(());
    let x_scale = TensorTrait::new(shape.span(), data.span(), extra);

    // ZEROPOINT
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(1);
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(84, false));
    data.append(IntegerTrait::new(24, false));
    data.append(IntegerTrait::new(196, false));
    let extra = Option::<ExtraParams>::None(());
    let x_zero_point = TensorTrait::new(shape.span(), data.span(), extra);

    let y: Tensor<i32> = x.dequantize_linear(@x_scale, @x_zero_point);

    assert((*y.data[0]).into() == -162, '*result[0] == -162');
    assert((*y.data[1]).into() == 10, '*result[1] == 10');
    assert((*y.data[2]).into() == -100, '*result[2] == -100');
    assert((*y.data[3]).into() == 232, '*result[3] == 232');
    assert((*y.data[4]).into() == -20, '*result[4] == -20');
    assert((*y.data[5]).into() == -50, '*result[5] == -50');
    assert((*y.data[6]).into() == -76, '*result[6] == -76');
    assert((*y.data[7]).into() == 0, '*result[7] == 0');
    assert((*y.data[8]).into() == 0, '*result[8] == 0');
    assert((*y.data[9]).into() == 252, '*result[9] == 252');
    assert((*y.data[10]).into() == 32, '*result[10] == 32');
    assert((*y.data[11]).into() == -44, '*result[11] == -44');
    assert((*y.data[12]).into() == 245, '*result[12] == 245');
    assert((*y.data[13]).into() == -485, '*result[13] == -485');
    assert((*y.data[14]).into() == -960, '*result[14] == -960');
    assert((*y.data[15]).into() == -270, '*result[15] == -270');
    assert((*y.data[16]).into() == -375, '*result[16] == -375');
    assert((*y.data[17]).into() == -470, '*result[17] == -470');
}
