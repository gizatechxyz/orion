use array::ArrayTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::vector::sum_two_vec;
use onnx_cairo::operators::math::vector::find_max;
use onnx_cairo::operators::math::vector::find_min;
use onnx_cairo::operators::math::vector::sum_vec;
use onnx_cairo::operators::math::int33::i33;

// impl Arrayi33Drop of Drop::<Array::<i33>>;

#[test]
#[available_gas(2000000)]
fn sum_test() {
    // Test with random numbers

    let mut vec1 = ArrayTrait::new();
    vec1.append(i33 { inner: 23293_u32, sign: true });
    vec1.append(i33 { inner: 17752_u32, sign: false });
    vec1.append(i33 { inner: 32068_u32, sign: true });

    let mut vec2 = ArrayTrait::new();
    vec2.append(i33 { inner: 23533_u32, sign: false });
    vec2.append(i33 { inner: 24476_u32, sign: false });
    vec2.append(i33 { inner: 29585_u32, sign: true });

    let mut result = sum_two_vec(vec1, vec2);

    assert(*result.at(0_usize).inner == 240_u32, 'result[0] == 240');
    assert(*result.at(0_usize).sign == false, 'result[0] == 240');
    assert(*result.at(1_usize).inner == 42228_u32, 'result[1] == 42228');
    assert(*result.at(1_usize).sign == false, 'result[1] == 42228');
    assert(*result.at(2_usize).inner == 61653_u32, 'result[2] == -61653');
    assert(*result.at(2_usize).sign == true, 'result[2] == -61653');

    let mut vec1 = ArrayTrait::new();
    vec1.append(i33 { inner: 3243_u32, sign: false });
    vec1.append(i33 { inner: 29084_u32, sign: true });
    vec1.append(i33 { inner: 16650_u32, sign: false });

    let mut vec2 = ArrayTrait::new();
    vec2.append(i33 { inner: 28891_u32, sign: false });
    vec2.append(i33 { inner: 11856_u32, sign: true });
    vec2.append(i33 { inner: 30435_u32, sign: true });

    let mut result = sum_two_vec(vec1, vec2);

    assert(*result.at(0_usize).inner == 32134_u32, 'result[0] == 32134');
    assert(*result.at(0_usize).sign == false, 'result[0] == 32134');
    assert(*result.at(1_usize).inner == 40940_u32, 'result[1] == -40940');
    assert(*result.at(1_usize).sign == true, 'result[1] == -40940');
    assert(*result.at(2_usize).inner == 13785_u32, 'result[2] == -13785');
    assert(*result.at(2_usize).sign == true, 'result[2] == -13785');

    let mut vec1 = ArrayTrait::new();
    vec1.append(i33 { inner: 32767_u32, sign: false });
    vec1.append(i33 { inner: 32768_u32, sign: true });
    vec1.append(i33 { inner: 0_u32, sign: false });

    let mut vec2 = ArrayTrait::new();
    vec2.append(i33 { inner: 10000_u32, sign: false });
    vec2.append(i33 { inner: 15000_u32, sign: true });
    vec2.append(i33 { inner: 16384_u32, sign: false });

    let mut result = sum_two_vec(vec1, vec2);

    assert(*result.at(0_usize).inner == 42767_u32, 'result[0] == 42767');
    assert(*result.at(0_usize).sign == false, 'result[0] == 42767');
    assert(*result.at(1_usize).inner == 47768_u32, 'result[1] == -47768');
    assert(*result.at(1_usize).sign == true, 'result[1] == -47768');
    assert(*result.at(2_usize).inner == 16384_u32, 'result[2] == 16384');
    assert(*result.at(2_usize).sign == false, 'result[2] == 16384');
}

#[test]
#[available_gas(2000000)]
fn find_min_test() {
    let mut vec = ArrayTrait::new();
    vec.append(i33 { inner: 80_u32, sign: false });
    vec.append(i33 { inner: 80_u32, sign: true });
    vec.append(i33 { inner: 50_u32, sign: false });
    vec.append(i33 { inner: 50_u32, sign: true });
    vec.append(i33 { inner: 25_u32, sign: false });
    vec.append(i33 { inner: 25_u32, sign: true });
    vec.append(i33 { inner: 127_u32, sign: false });
    vec.append(i33 { inner: 128_u32, sign: true });

    let min = find_min(@vec);

    assert(min.inner == 128_u32 & min.sign == true, 'min: -128');
}

#[test]
#[available_gas(2000000)]
fn find_max_test() {
    let mut vec = ArrayTrait::new();
    vec.append(i33 { inner: 80_u32, sign: false });
    vec.append(i33 { inner: 80_u32, sign: true });
    vec.append(i33 { inner: 50_u32, sign: false });
    vec.append(i33 { inner: 50_u32, sign: true });
    vec.append(i33 { inner: 25_u32, sign: false });
    vec.append(i33 { inner: 25_u32, sign: true });
    vec.append(i33 { inner: 127_u32, sign: false });
    vec.append(i33 { inner: 128_u32, sign: true });

    let max = find_max(@vec);

    assert(max.inner == 127_u32 & max.sign == false, 'max: 127');
}

#[test]
#[available_gas(2000000)]
fn sum_vec_test() {
    let mut vec = ArrayTrait::new();
    vec.append(i33 { inner: 80_u32, sign: false });
    vec.append(i33 { inner: 80_u32, sign: true });
    vec.append(i33 { inner: 50_u32, sign: false });

    let min = sum_vec(@vec);

    assert(min.inner == 50_u32 & min.sign == false, 'sum: 50');
}
