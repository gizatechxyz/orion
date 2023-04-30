use array::ArrayTrait;

use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32;
use onnx_cairo::performance::quantizations::quant_vec;


#[test]
#[available_gas(2000000)]
fn quant_vec_test() {
    let mut vec = ArrayTrait::<i32>::new();
    vec.append(IntegerTrait::new(30523_u32, true));
    vec.append(IntegerTrait::new(24327_u32, false));
    vec.append(IntegerTrait::new(12288_u32, true));
    vec.append(IntegerTrait::new(29837_u32, false));
    vec.append(IntegerTrait::new(19345_u32, true));
    vec.append(IntegerTrait::new(15416_u32, false));

    let mut res = quant_vec(@vec);

    assert(*res.at(0_usize).mag == 127_u32, '*result[0] == -127');
    assert(*res.at(0_usize).sign == true, '*result[0] -> negative');

    assert(*res.at(1_usize).mag == 101_u32, '*result[1] == 101');
    assert(*res.at(1_usize).sign == false, '*result[1] -> positive');

    assert(*res.at(2_usize).mag == 51_u32, '*result[2] == -51');
    assert(*res.at(2_usize).sign == true, '*result[2] -> negative');

    assert(*res.at(3_usize).mag == 124_u32, '*result[3] == 124');
    assert(*res.at(3_usize).sign == false, '*result[3] -> positive');

    assert(*res.at(4_usize).mag == 80_u32, '*result[4] == -80');
    assert(*res.at(4_usize).sign == true, '*result[4] -> negative');

    assert(*res.at(5_usize).mag == 64_u32, '*result[5] == 64');
    assert(*res.at(5_usize).sign == false, '*result[5] -> positive');
}
