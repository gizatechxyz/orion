use core::debug::{PrintTrait};
use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::{TensorTrait, ExtraParams, Tensor};
use orion::performance::core::PerfomanceTrait;
use orion::performance::implementations::impl_performance_u32::Performance_u32;

#[test]
#[available_gas(2000000)]
fn dequantize_linear() {
    // X
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);
    let mut data = ArrayTrait::<u32>::new();
    data.append(128);
    data.append(129);
    data.append(129);
    data.append(255);
    data.append(255);
    data.append(255);
    let extra = Option::<ExtraParams>::None(());
    let x = TensorTrait::new(shape.span(), data.span(), extra);

    // XSCALE
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    let mut data = ArrayTrait::<u32>::new();
    data.append(2);
    let extra = Option::<ExtraParams>::None(());
    let x_scale = TensorTrait::new(shape.span(), data.span(), extra);

    // ZEROPOINT
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    let mut data = ArrayTrait::<u32>::new();
    data.append(128);
    let extra = Option::<ExtraParams>::None(());
    let x_zero_point = TensorTrait::new(shape.span(), data.span(), extra);

    let y: Tensor<u32> = x.dequantize_linear(@x_scale, @x_zero_point);

    assert((*y.data[0]).into() == 0, '*result[0] == 0');
    assert((*y.data[1]).into() == 2, '*result[1] == 2');
    assert((*y.data[2]).into() == 2, '*result[2] == 2');
    assert((*y.data[3]).into() == 254, '*result[3] == 254');
    assert((*y.data[4]).into() == 254, '*result[4] == 254');
    assert((*y.data[5]).into() == 254, '*result[5] == 254');
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
    let mut data = ArrayTrait::<u32>::new();
    data.append(165);
    data.append(89);
    data.append(134);
    data.append(200);
    data.append(94);
    data.append(109);
    data.append(43);
    data.append(24);
    data.append(24);
    data.append(87);
    data.append(32);
    data.append(35);
    data.append(245);
    data.append(255);
    data.append(255);
    data.append(250);
    data.append(255);
    data.append(255);
    let extra = Option::<ExtraParams>::None(());
    let x = TensorTrait::new(shape.span(), data.span(), extra);

    // XSCALE
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(1);
    shape.append(1);
    let mut data = ArrayTrait::<u32>::new();
    data.append(2);
    data.append(4);
    data.append(5);
    let extra = Option::<ExtraParams>::None(());
    let x_scale = TensorTrait::new(shape.span(), data.span(), extra);

    // ZEROPOINT
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(1);
    shape.append(1);
    let mut data = ArrayTrait::<u32>::new();
    data.append(84);
    data.append(24);
    data.append(196);
    let extra = Option::<ExtraParams>::None(());
    let x_zero_point = TensorTrait::new(shape.span(), data.span(), extra);

    let y: Tensor<u32> = x.dequantize_linear(@x_scale, @x_zero_point);

    assert((*y.data[0]).into() == 162, '*result[0] == 162');
    assert((*y.data[1]).into() == 10, '*result[1] == 10');
    assert((*y.data[2]).into() == 100, '*result[2] == 100');
    assert((*y.data[3]).into() == 232, '*result[3] == 232');
    assert((*y.data[4]).into() == 20, '*result[4] == 20');
    assert((*y.data[5]).into() == 50, '*result[5] == 50');
    assert((*y.data[6]).into() == 76, '*result[6] == 76');
    assert((*y.data[7]).into() == 0, '*result[7] == 0');
    assert((*y.data[8]).into() == 0, '*result[8] == 0');
    assert((*y.data[9]).into() == 252, '*result[9] == 252');
    assert((*y.data[10]).into() == 32, '*result[10] == 32');
    assert((*y.data[11]).into() == 44, '*result[11] == 44');
    assert((*y.data[12]).into() == 245, '*result[12] == 245');
    assert((*y.data[13]).into() == 295, '*result[13] == 295');
    assert((*y.data[14]).into() == 295, '*result[14] == 295');
    assert((*y.data[15]).into() == 270, '*result[15] == 270');
    assert((*y.data[16]).into() == 295, '*result[16] == 295');
    assert((*y.data[17]).into() == 295, '*result[17] == 295');
}

