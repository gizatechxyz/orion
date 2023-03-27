use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::matrix::Matrix;
use onnx_cairo::operators::math::matrix::MatrixTrait;
use onnx_cairo::operators::math::int33;
use onnx_cairo::operators::math::int33::i33;


#[test]
#[available_gas(2000000)]
fn matrix_new_test() {
    let mut arr = ArrayTrait::<i33>::new();
    let val_1 = i33 { inner: 1_u32, sign: false };
    let val_2 = i33 { inner: 2_u32, sign: false };
    let val_3 = i33 { inner: 3_u32, sign: false };
    let val_4 = i33 { inner: 4_u32, sign: false };

    arr.append(val_1);
    arr.append(val_2);
    arr.append(val_3);
    arr.append(val_4);

    let mut matrix = MatrixTrait::new(2_usize, 2_usize, arr);
    let result_len = matrix.len();

    assert(result_len == 4_usize, 'correct length');
}

#[test]
#[available_gas(2000000)]
fn matrix_get_test() {
    let mut arr = ArrayTrait::<i33>::new();
    let val_1 = i33 { inner: 1_u32, sign: false };
    let val_2 = i33 { inner: 2_u32, sign: true };
    let val_3 = i33 { inner: 3_u32, sign: false };
    let val_4 = i33 { inner: 4_u32, sign: true };

    arr.append(val_1);
    arr.append(val_2);
    arr.append(val_3);
    arr.append(val_4);

    let mut matrix = MatrixTrait::new(2_usize, 2_usize, arr);
    let result = matrix.get(0_usize, 0_usize);

    assert(result.inner == 1_u32, 'result[0] == -1');
    assert(result.sign == false, 'result[0] == -1');

    let result = matrix.get(1_usize, 1_usize);

    assert(result.inner == 4_u32, 'result[8] == -4');
    assert(result.sign == true, 'result[8] == -4');
}

#[test]
#[available_gas(2000000)]
fn dot_test() {
    // Test with random numbers

    let mut data = ArrayTrait::new();
    data.append(i33 { inner: 87_u32, sign: false });
    data.append(i33 { inner: 28_u32, sign: true });
    data.append(i33 { inner: 104_u32, sign: true });
    data.append(i33 { inner: 42_u32, sign: false });
    data.append(i33 { inner: 6_u32, sign: true });
    data.append(i33 { inner: 75_u32, sign: false });

    let matrix = MatrixTrait::new(2_usize, 3_usize, data);

    let mut vec = ArrayTrait::new();
    vec.append(i33 { inner: 3_u32, sign: false });
    vec.append(i33 { inner: 63_u32, sign: true });
    vec.append(i33 { inner: 31_u32, sign: false });

    let new_matrix = MatrixTrait::new(3_usize, 1_usize, vec);

    let mut result = matrix.dot(@new_matrix);

    assert(result.len() == 2_usize, 'result.len() == 2');

    let data_0 = result.get(0_usize, 0_usize);
    assert(data_0.inner == 1199_u32, 'result[0] == -1199');
    assert(data_0.sign == true, 'result[0] == -1199');

    let data_1 = result.get(1_usize, 0_usize);
    assert(data_1.inner == 2829_u32, 'result[0] == 2829');
    assert(data_1.sign == false, 'result[0] == 2829');
}


#[test]
#[available_gas(2000000)]
fn add_test() {
    // Test with random numbers

    let mut vec = ArrayTrait::new();
    vec.append(i33 { inner: 1_u32, sign: false });
    vec.append(i33 { inner: 2_u32, sign: false });
    vec.append(i33 { inner: 3_u32, sign: false });
    vec.append(i33 { inner: 4_u32, sign: false });

    let matrix = MatrixTrait::new(2_usize, 2_usize, vec);

    let mut new_vec = ArrayTrait::new();
    new_vec.append(i33 { inner: 1_u32, sign: false });
    new_vec.append(i33 { inner: 2_u32, sign: false });
    new_vec.append(i33 { inner: 3_u32, sign: false });
    new_vec.append(i33 { inner: 4_u32, sign: false });

    let new_matrix = MatrixTrait::new(2_usize, 2_usize, new_vec);

    let mut result = matrix.add(@new_matrix);

    assert(result.len() == 4_usize, 'result.len() == 4');

    let data_0 = result.get(0_usize, 0_usize);
    assert(data_0.inner == 2_u32, 'result[0] == 2');
    assert(data_0.sign == false, 'result[0] == 2');

    let data_1 = result.get(0_usize, 1_usize);
    assert(data_1.inner == 4_u32, 'result[1] == 4');
    assert(data_1.sign == false, 'result[1] == 4');

    let data_2 = result.get(1_usize, 0_usize);
    assert(data_2.inner == 6_u32, 'result[2] == 6');
    assert(data_2.sign == false, 'result[2] == 6');

    let data_3 = result.get(1_usize, 1_usize);
    assert(data_3.inner == 8_u32, 'result[3] == 8');
    assert(data_3.sign == false, 'result[3] == 8');
}


#[test]
#[available_gas(2000000)]
fn argmax_test() {
    // Test with random numbers

    let mut vec = ArrayTrait::new();
    vec.append(i33 { inner: 1_u32, sign: false });
    vec.append(i33 { inner: 2_u32, sign: false });
    vec.append(i33 { inner: 3_u32, sign: false });
    vec.append(i33 { inner: 4_u32, sign: false });

    let matrix = MatrixTrait::new(2_usize, 2_usize, vec);
    let mut result = matrix.argmax();

    assert(*result.at(0_usize) == 1_usize, 'row 0 max index: 1');
    assert(*result.at(1_usize) == 1_usize, 'row 1 max index: 1');
}
