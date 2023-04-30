use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::matrix::Matrix;
use onnx_cairo::operators::math::matrix::MatrixTrait;
use onnx_cairo::operators::math::signed_integer;
use onnx_cairo::operators::math::signed_integer::i32;


#[test]
#[available_gas(2000000)]
fn matrix_new_test() {
    let mut arr = ArrayTrait::<i32>::new();
    let val_1 = IntegerTrait::new(1_u32, false);
    let val_2 = IntegerTrait::new(2_u32, false);
    let val_3 = IntegerTrait::new(3_u32, false);
    let val_4 = IntegerTrait::new(4_u32, false);

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
    let mut arr = ArrayTrait::<i32>::new();
    let val_1 = IntegerTrait::new(1_u32, false);
    let val_2 = IntegerTrait::new(2_u32, true);
    let val_3 = IntegerTrait::new(3_u32, false);
    let val_4 = IntegerTrait::new(4_u32, true);

    arr.append(val_1);
    arr.append(val_2);
    arr.append(val_3);
    arr.append(val_4);

    let mut matrix = MatrixTrait::new(2_usize, 2_usize, arr);
    let result = matrix.get(0_usize, 0_usize);

    assert(result.mag == 1_u32, 'result[0] == -1');
    assert(result.sign == false, 'result[0] == -1');

    let result = matrix.get(1_usize, 1_usize);

    assert(result.mag == 4_u32, 'result[8] == -4');
    assert(result.sign == true, 'result[8] == -4');
}

#[test]
#[available_gas(2000000)]
fn dot_test() {
    // Test with random numbers

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(87_u32, false));
    data.append(IntegerTrait::new(28_u32, true));
    data.append(IntegerTrait::new(104_u32, true));
    data.append(IntegerTrait::new(42_u32, false));
    data.append(IntegerTrait::new(6_u32, true));
    data.append(IntegerTrait::new(75_u32, false));

    let matrix = MatrixTrait::new(2_usize, 3_usize, data);

    let mut vec = ArrayTrait::new();
    vec.append(IntegerTrait::new(3_u32, false));
    vec.append(IntegerTrait::new(63_u32, true));
    vec.append(IntegerTrait::new(31_u32, false));

    let new_matrix = MatrixTrait::new(3_usize, 1_usize, vec);

    let mut result = matrix.dot(@new_matrix);

    assert(result.len() == 2_usize, 'result.len() == 2');

    let data_0 = result.get(0_usize, 0_usize);
    assert(data_0.mag == 1199_u32, 'result[0] == -1199');
    assert(data_0.sign == true, 'result[0] == -1199');

    let data_1 = result.get(1_usize, 0_usize);
    assert(data_1.mag == 2829_u32, 'result[0] == 2829');
    assert(data_1.sign == false, 'result[0] == 2829');
}


#[test]
#[available_gas(2000000)]
fn add_test() {
    // Test with random numbers

    let mut vec = ArrayTrait::new();
    vec.append(IntegerTrait::new(1_u32, false));
    vec.append(IntegerTrait::new(2_u32, false));
    vec.append(IntegerTrait::new(3_u32, false));
    vec.append(IntegerTrait::new(4_u32, false));

    let matrix = MatrixTrait::new(2_usize, 2_usize, vec);

    let mut new_vec = ArrayTrait::new();
    new_vec.append(IntegerTrait::new(1_u32, false));
    new_vec.append(IntegerTrait::new(2_u32, false));
    new_vec.append(IntegerTrait::new(3_u32, false));
    new_vec.append(IntegerTrait::new(4_u32, false));

    let new_matrix = MatrixTrait::new(2_usize, 2_usize, new_vec);

    let mut result = matrix.add(@new_matrix);

    assert(result.len() == 4_usize, 'result.len() == 4');

    let data_0 = result.get(0_usize, 0_usize);
    assert(data_0.mag == 2_u32, 'result[0] == 2');
    assert(data_0.sign == false, 'result[0] == 2');

    let data_1 = result.get(0_usize, 1_usize);
    assert(data_1.mag == 4_u32, 'result[1] == 4');
    assert(data_1.sign == false, 'result[1] == 4');

    let data_2 = result.get(1_usize, 0_usize);
    assert(data_2.mag == 6_u32, 'result[2] == 6');
    assert(data_2.sign == false, 'result[2] == 6');

    let data_3 = result.get(1_usize, 1_usize);
    assert(data_3.mag == 8_u32, 'result[3] == 8');
    assert(data_3.sign == false, 'result[3] == 8');
}


#[test]
#[available_gas(2000000)]
fn argmax_test() {
    // Test with random numbers

    let mut vec = ArrayTrait::new();
    vec.append(IntegerTrait::new(1_u32, false));
    vec.append(IntegerTrait::new(2_u32, false));
    vec.append(IntegerTrait::new(3_u32, false));
    vec.append(IntegerTrait::new(4_u32, false));

    let matrix = MatrixTrait::new(2_usize, 2_usize, vec);
    let mut result = matrix.argmax();

    assert(*result.at(0_usize) == 1_usize, 'row 0 max index: 1');
    assert(*result.at(1_usize) == 1_usize, 'row 1 max index: 1');
}

#[test]
#[available_gas(2000000)]
fn reduce_sum_test() {
    // Test with random numbers

    let mut vec = ArrayTrait::new();
    vec.append(IntegerTrait::new(1_u32, false));
    vec.append(IntegerTrait::new(2_u32, false));
    vec.append(IntegerTrait::new(3_u32, false));
    vec.append(IntegerTrait::new(4_u32, false));

    let matrix = MatrixTrait::new(2_usize, 2_usize, vec);
    let mut result = matrix.reduce_sum();

    assert(result.mag == 10_usize, '10');
    assert(result.sign == false, '10');
}

