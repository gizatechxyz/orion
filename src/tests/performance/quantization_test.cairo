use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::performance::core::PerfomanceTrait;
use orion::performance::implementations::impl_performance_i32::Performance_i32;


#[test]
#[available_gas(2000000)]
fn quantize_linear_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(30523, true));
    data.append(IntegerTrait::new(24327, false));
    data.append(IntegerTrait::new(12288, true));
    data.append(IntegerTrait::new(29837, false));
    data.append(IntegerTrait::new(19345, true));
    data.append(IntegerTrait::new(15416, false));
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::new(shape.span(), data.span(), extra);

    let mut res = PerfomanceTrait::quantize_linear(@tensor);

    assert((*res.data[0]).into() == -127, '*result[0] == -127');
    assert((*res.data[1]).into() == 101, '*result[1] == 101');
    assert((*res.data[2]).into() == -51, '*result[2] == -51');
    assert((*res.data[3]).into() == 124, '*result[3] == 124');
    assert((*res.data[4]).into() == -80, '*result[4] == -80');
    assert((*res.data[5]).into() == 64, '*result[5] == 64');
}

#[test]
#[available_gas(2000000)]
fn quantize_linear_from_fp_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::<FixedType>::new();
    data.append(FixedTrait::new(838860800, false)); // 100
    data.append(FixedTrait::new(1258291200, false)); // 150
    data.append(FixedTrait::new(1677721600, false)); // 200
    data.append(FixedTrait::new(838860800, true)); // -100
    data.append(FixedTrait::new(1258291200, true)); // -150
    data.append(FixedTrait::new(1677721600, true)); // -200

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(shape.span(), data.span(), Option::Some(extra));

    let mut res = PerfomanceTrait::<i32>::quantize_linear_from_fp(@tensor);

    assert((*res.data[0]).into() == 63, '*result[0] == 63');
    assert((*res.data[1]).into() == 95, '*result[1] == 95');
    assert((*res.data[2]).into() == 127, '*result[2] == 127');
    assert((*res.data[3]).into() == -63, '*result[3] == -63');
    assert((*res.data[4]).into() == -95, '*result[4] == -95');
    assert((*res.data[5]).into() == -127, '*result[5] == -127');
}
