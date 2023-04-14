use array::ArrayTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::vector::sum_two_vec;
use onnx_cairo::operators::math::vector::find_max;
use onnx_cairo::operators::math::vector::find_min;
use onnx_cairo::operators::math::vector::sum_vec;
use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32;

// impl Arrayi32Drop of Drop::<Array::<i32>>;

#[test]
#[available_gas(2000000)]
fn sum_test() {
    // Test with random numbers

    let mut vec1 = ArrayTrait::new();
    vec1.append(IntegerTrait::new(23293_u32, true));
    vec1.append(IntegerTrait::new(17752_u32, false));
    vec1.append(IntegerTrait::new(32068_u32, true));

    let mut vec2 = ArrayTrait::new();
    vec2.append(IntegerTrait::new(23533_u32, false));
    vec2.append(IntegerTrait::new(24476_u32, false));
    vec2.append(IntegerTrait::new(29585_u32, true));

    let mut result = sum_two_vec(vec1, vec2);

    assert(*result.at(0_usize).mag == 240_u32, 'result[0] == 240');
    assert(*result.at(0_usize).sign == false, 'result[0] == 240');
    assert(*result.at(1_usize).mag == 42228_u32, 'result[1] == 42228');
    assert(*result.at(1_usize).sign == false, 'result[1] == 42228');
    assert(*result.at(2_usize).mag == 61653_u32, 'result[2] == -61653');
    assert(*result.at(2_usize).sign == true, 'result[2] == -61653');

    let mut vec1 = ArrayTrait::new();
    vec1.append(IntegerTrait::new(3243_u32, false));
    vec1.append(IntegerTrait::new(29084_u32, true));
    vec1.append(IntegerTrait::new(16650_u32, false));

    let mut vec2 = ArrayTrait::new();
    vec2.append(IntegerTrait::new(28891_u32, false));
    vec2.append(IntegerTrait::new(11856_u32, true));
    vec2.append(IntegerTrait::new(30435_u32, true));

    let mut result = sum_two_vec(vec1, vec2);

    assert(*result.at(0_usize).mag == 32134_u32, 'result[0] == 32134');
    assert(*result.at(0_usize).sign == false, 'result[0] == 32134');
    assert(*result.at(1_usize).mag == 40940_u32, 'result[1] == -40940');
    assert(*result.at(1_usize).sign == true, 'result[1] == -40940');
    assert(*result.at(2_usize).mag == 13785_u32, 'result[2] == -13785');
    assert(*result.at(2_usize).sign == true, 'result[2] == -13785');

    let mut vec1 = ArrayTrait::new();
    vec1.append(IntegerTrait::new(32767_u32, false));
    vec1.append(IntegerTrait::new(32768_u32, true));
    vec1.append(IntegerTrait::new(0_u32, false));

    let mut vec2 = ArrayTrait::new();
    vec2.append(IntegerTrait::new(10000_u32, false));
    vec2.append(IntegerTrait::new(15000_u32, true));
    vec2.append(IntegerTrait::new(16384_u32, false));

    let mut result = sum_two_vec(vec1, vec2);

    assert(*result.at(0_usize).mag == 42767_u32, 'result[0] == 42767');
    assert(*result.at(0_usize).sign == false, 'result[0] == 42767');
    assert(*result.at(1_usize).mag == 47768_u32, 'result[1] == -47768');
    assert(*result.at(1_usize).sign == true, 'result[1] == -47768');
    assert(*result.at(2_usize).mag == 16384_u32, 'result[2] == 16384');
    assert(*result.at(2_usize).sign == false, 'result[2] == 16384');
}

#[test]
#[available_gas(2000000)]
fn find_min_test() {
    let mut vec = ArrayTrait::new();
    vec.append(IntegerTrait::new(80_u32, false));
    vec.append(IntegerTrait::new(80_u32, true));
    vec.append(IntegerTrait::new(50_u32, false));
    vec.append(IntegerTrait::new(50_u32, true));
    vec.append(IntegerTrait::new(25_u32, false));
    vec.append(IntegerTrait::new(25_u32, true));
    vec.append(IntegerTrait::new(127_u32, false));
    vec.append(IntegerTrait::new(128_u32, true));

    let min = find_min(@vec);

    assert(min.mag == 128_u32 & min.sign == true, 'min: -128');
}

#[test]
#[available_gas(2000000)]
fn find_max_test() {
    let mut vec = ArrayTrait::new();
    vec.append(IntegerTrait::new(80_u32, false));
    vec.append(IntegerTrait::new(80_u32, true));
    vec.append(IntegerTrait::new(50_u32, false));
    vec.append(IntegerTrait::new(50_u32, true));
    vec.append(IntegerTrait::new(25_u32, false));
    vec.append(IntegerTrait::new(25_u32, true));
    vec.append(IntegerTrait::new(127_u32, false));
    vec.append(IntegerTrait::new(128_u32, true));

    let max = find_max(@vec);

    assert(max.mag == 127_u32 & max.sign == false, 'max: 127');
}

#[test]
#[available_gas(2000000)]
fn sum_vec_test() {
    let mut vec = ArrayTrait::new();
    vec.append(IntegerTrait::new(80_u32, false));
    vec.append(IntegerTrait::new(80_u32, true));
    vec.append(IntegerTrait::new(50_u32, false));

    let min = sum_vec(@vec);

    assert(min.mag == 50_u32 & min.sign == false, 'sum: 50');
}
