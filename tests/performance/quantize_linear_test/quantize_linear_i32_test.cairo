use core::debug::{PrintTrait};
use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::traits::Into;

use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::I32Tensor;
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000)]
fn quantize_linear() {
    // X
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);
    let mut data = ArrayTrait::<i32>::new();
    data.append(0);
    data.append(2);
    data.append(3);
    data.append(1000);
    data.append(-254);
    data.append(-1000);

    let x = TensorTrait::new(shape.span(), data.span());

    // YSCALE
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(2);

    let y_scale = TensorTrait::new(shape.span(), data.span());

    // ZEROPOINT
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(1);

    let y_zero_point = TensorTrait::new(shape.span(), data.span());

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
    let mut data = ArrayTrait::<i32>::new();
    data.append(-162);
    data.append(10);
    data.append(-100);
    data.append(232);
    data.append(-20);
    data.append(-50);
    data.append(-76);
    data.append(0);
    data.append(0);
    data.append(252);
    data.append(32);
    data.append(-44);
    data.append(245);
    data.append(-485);
    data.append(-960);
    data.append(-270);
    data.append(-375);
    data.append(-470);

    let x = TensorTrait::new(shape.span(), data.span());

    // YSCALE
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(1);
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(2);
    data.append(4);
    data.append(5);

    let y_scale = TensorTrait::new(shape.span(), data.span());

    // ZEROPOINT
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(1);
    shape.append(1);
    let mut data = ArrayTrait::<i32>::new();
    data.append(84);
    data.append(24);
    data.append(196);

    let y_zero_point = TensorTrait::new(shape.span(), data.span());

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
