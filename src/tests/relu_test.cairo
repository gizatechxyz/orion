use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::matrix::Matrix;
use onnx_cairo::operators::math::matrix::MatrixTrait;
use onnx_cairo::operators::math::int33;
use onnx_cairo::operators::math::int33::i33;
use onnx_cairo::operators::activations::relu::relu;

// impl Arrayi33Drop of Drop::<Array::<i33>>;

#[test]
#[available_gas(2000000)]
fn relu_test() {
    let mut arr = ArrayTrait::<i33>::new();
    let val_1 = i33 { inner: 1_u32, sign: false };
    let val_2 = i33 { inner: 2_u32, sign: false };
    let val_3 = i33 { inner: 1_u32, sign: true };
    let val_4 = i33 { inner: 2_u32, sign: true };

    arr.append(val_1);
    arr.append(val_2);
    arr.append(val_3);
    arr.append(val_4);

    let mut matrix = MatrixTrait::new(2_usize, 2_usize, arr);
    let mut result_matrix = relu(@matrix);

    let data_0 = result_matrix.get(0_usize, 0_usize);
    assert(data_0.inner == 1_u32, 'result[0] == 1');
    assert(data_0.sign == false, 'result[0] == 1');

    let data_4 = result_matrix.get(1_usize, 1_usize);
    assert(data_4.inner == 0_u32, 'result[4] == 0');
    assert(data_4.sign == true, 'result[4] == 0');
}
