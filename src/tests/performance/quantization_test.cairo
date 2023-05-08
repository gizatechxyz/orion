use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use onnx_cairo::operators::tensor::implementations::impl_tensor_i32;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::performance::performance_i32::performance::quantize_linear;

#[test]
#[available_gas(2000000)]
fn quant_vec_test() {
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

    let mut res = quantize_linear(@tensor);

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
