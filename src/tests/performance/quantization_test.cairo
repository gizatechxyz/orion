use array::ArrayTrait;
use array::SpanTrait;
use debug::PrintTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::numbers::fixed_point::types::{Fixed, FixedType};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::operators::tensor::core::TensorTrait;
use orion::performance::core::PerfomanceTrait;
use orion::performance::implementations::impl_performance_i32;


#[test]
#[available_gas(2000000)]
fn quantize_linear_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::<i32>::new();
    data.append(IntegerTrait::new(30523_u32, true));
    data.append(IntegerTrait::new(24327_u32, false));
    data.append(IntegerTrait::new(12288_u32, true));
    data.append(IntegerTrait::new(29837_u32, false));
    data.append(IntegerTrait::new(19345_u32, true));
    data.append(IntegerTrait::new(15416_u32, false));

    let tensor = TensorTrait::new(shape.span(), data.span());

    let mut res = PerfomanceTrait::quantize_linear(@tensor);

    assert(*res.data.at(0_usize).mag == 127_u32, '*result[0] == -127');
    assert(*res.data.at(0_usize).sign == true, '*result[0] -> negative');

    assert(*res.data.at(1_usize).mag == 101_u32, '*result[1] == 101');
    assert(*res.data.at(1_usize).sign == false, '*result[1] -> positive');

    assert(*res.data.at(2_usize).mag == 51_u32, '*result[2] == -51');
    assert(*res.data.at(2_usize).sign == true, '*result[2] -> negative');

    assert(*res.data.at(3_usize).mag == 124_u32, '*result[3] == 124');
    assert(*res.data.at(3_usize).sign == false, '*result[3] -> positive');

    assert(*res.data.at(4_usize).mag == 80_u32, '*result[4] == -80');
    assert(*res.data.at(4_usize).sign == true, '*result[4] -> negative');

    assert(*res.data.at(5_usize).mag == 64_u32, '*result[5] == 64');
    assert(*res.data.at(5_usize).sign == false, '*result[5] -> positive');
}

#[test]
#[available_gas(2000000)]
fn quantize_linear_from_fp_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::<FixedType>::new();
    data.append(Fixed::new(838860800, false)); // 100
    data.append(Fixed::new(1258291200, false)); // 150
    data.append(Fixed::new(1677721600, false)); // 200
    data.append(Fixed::new(838860800, true)); // -100
    data.append(Fixed::new(1258291200, true)); // -150
    data.append(Fixed::new(1677721600, true)); // -200

    let tensor = TensorTrait::<FixedType>::new(shape.span(), data.span());

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
