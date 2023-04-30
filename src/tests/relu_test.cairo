use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::matrix::Matrix;
use onnx_cairo::operators::math::matrix::MatrixTrait;
use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32;
use onnx_cairo::operators::activations::relu::relu;

// impl Arrayi32Drop of Drop::<Array::<i32>>;

#[test]
#[available_gas(2000000)]
fn relu_test() {
    let mut arr = ArrayTrait::<i32>::new();
    let val_1 = IntegerTrait::new(1_u32, false);
    let val_2 = IntegerTrait::new(2_u32, false);
    let val_3 = IntegerTrait::new(1_u32, true);
    let val_4 = IntegerTrait::new(2_u32, true);

    arr.append(val_1);
    arr.append(val_2);
    arr.append(val_3);
    arr.append(val_4);

    let mut matrix = MatrixTrait::new(2_usize, 2_usize, arr);
    let mut result_matrix = relu(@matrix);

    let data_0 = result_matrix.get(0_usize, 0_usize);
    assert(data_0.mag == 1_u32, 'result[0] == 1');
    assert(data_0.sign == false, 'result[0] == 1');

    let data_4 = result_matrix.get(1_usize, 1_usize);
    assert(data_4.mag == 0_u32, 'result[4] == 0');
    assert(data_4.sign == false, 'result[4] == 0');
}
