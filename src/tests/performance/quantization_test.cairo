use array::ArrayTrait;
use array::SpanTrait;

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

    assert(*res.data.at(0_usize).mag == 127, '*result[0] == -127');
    assert(*res.data.at(0_usize).sign == true, '*result[0] -> negative');

    assert(*res.data.at(1_usize).mag == 101, '*result[1] == 101');
    assert(*res.data.at(1_usize).sign == false, '*result[1] -> positive');

    assert(*res.data.at(2_usize).mag == 51, '*result[2] == -51');
    assert(*res.data.at(2_usize).sign == true, '*result[2] -> negative');

    assert(*res.data.at(3_usize).mag == 124, '*result[3] == 124');
    assert(*res.data.at(3_usize).sign == false, '*result[3] -> positive');

    assert(*res.data.at(4_usize).mag == 80, '*result[4] == -80');
    assert(*res.data.at(4_usize).sign == true, '*result[4] -> negative');

    assert(*res.data.at(5_usize).mag == 64, '*result[5] == 64');
    assert(*res.data.at(5_usize).sign == false, '*result[5] -> positive');
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

    assert(*res.data.at(0_usize).mag == 63, '*result[0] == 63');
    assert(*res.data.at(0_usize).sign == false, '*result[0] -> positive');

    assert(*res.data.at(1_usize).mag == 95, '*result[1] == 95');
    assert(*res.data.at(1_usize).sign == false, '*result[1] -> positive');

    assert(*res.data.at(2_usize).mag == 127, '*result[2] == 127');
    assert(*res.data.at(2_usize).sign == false, '*result[2] -> positive');

    assert(*res.data.at(3_usize).mag == 63, '*result[3] == -63');
    assert(*res.data.at(3_usize).sign == true, '*result[3] -> negative');

    assert(*res.data.at(4_usize).mag == 95, '*result[4] == -95');
    assert(*res.data.at(4_usize).sign == true, '*result[4] -> negative');

    assert(*res.data.at(5_usize).mag == 127, '*result[5] == -127');
    assert(*res.data.at(5_usize).sign == true, '*result[5] -> negative');
}
