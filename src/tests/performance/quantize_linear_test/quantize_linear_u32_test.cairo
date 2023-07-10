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
fn quantize_linear() {
    // X
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);
    let mut data = ArrayTrait::<u32>::new();
    data.append(0);
    data.append(2);
    data.append(3);
    data.append(1000);
    data.append(254);
    data.append(1000);
    let extra = Option::<ExtraParams>::None(());
    let x = TensorTrait::new(shape.span(), data.span(), extra);

    // YSCALE
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    let mut data = ArrayTrait::<u32>::new();
    data.append(2);
    let extra = Option::<ExtraParams>::None(());
    let y_scale = TensorTrait::new(shape.span(), data.span(), extra);

    // ZEROPOINT
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    let mut data = ArrayTrait::<u32>::new();
    data.append(128);
    let extra = Option::<ExtraParams>::None(());
    let y_zero_point = TensorTrait::new(shape.span(), data.span(), extra);

    let y: Tensor<u32> = x.quantize_linear(@y_scale, @y_zero_point);

    assert((*y.data[0]).into() == 128, '*result[0] == 128');
    assert((*y.data[1]).into() == 129, '*result[1] == 129');
    assert((*y.data[2]).into() == 129, '*result[2] == 129');
    assert((*y.data[3]).into() == 255, '*result[3] == 255');
    assert((*y.data[4]).into() == 255, '*result[4] == 255');
    assert((*y.data[5]).into() == 255, '*result[5] == 255');
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
    data.append(162);
    data.append(10);
    data.append(100);
    data.append(232);
    data.append(20);
    data.append(50);
    data.append(76);
    data.append(0);
    data.append(0);
    data.append(252);
    data.append(32);
    data.append(44);
    data.append(245);
    data.append(485);
    data.append(960);
    data.append(270);
    data.append(375);
    data.append(470);
    let extra = Option::<ExtraParams>::None(());
    let x = TensorTrait::new(shape.span(), data.span(), extra);

    // YSCALE
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
    let y_scale = TensorTrait::new(shape.span(), data.span(), extra);

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
    let y_zero_point = TensorTrait::new(shape.span(), data.span(), extra);

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

