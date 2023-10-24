use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{
    TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor, FP32x32Tensor
};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};
use orion::numbers::{NumberTrait, IntegerTrait};
use orion::numbers::{i8, i32};


#[test]
#[available_gas(200000000000)]
fn qlinearmatmul_2D_test() {
    let a = TensorTrait::<i8>::new(
        shape: array![2, 4].span(),
        data: array![
            IntegerTrait::<i8>::new(1_u8, true),
            IntegerTrait::<i8>::new(2_u8, true),
            IntegerTrait::<i8>::new(3_u8, true),
            IntegerTrait::<i8>::new(4_u8, true),
            IntegerTrait::<i8>::new(5_u8, true),
            IntegerTrait::<i8>::new(6_u8, true),
            IntegerTrait::<i8>::new(7_u8, true),
            IntegerTrait::<i8>::new(8_u8, true)
        ]
            .span(),
    );
    let b = TensorTrait::<i8>::new(
        shape: array![4, 3].span(),
        data: array![
            IntegerTrait::<i8>::new(2_u8, true),
            IntegerTrait::<i8>::new(4_u8, true),
            IntegerTrait::<i8>::new(6_u8, true),
            IntegerTrait::<i8>::new(8_u8, true),
            IntegerTrait::<i8>::new(10_u8, true),
            IntegerTrait::<i8>::new(12_u8, true),
            IntegerTrait::<i8>::new(14_u8, true),
            IntegerTrait::<i8>::new(16_u8, true),
            IntegerTrait::<i8>::new(18_u8, true),
            IntegerTrait::<i8>::new(20_u8, true),
            IntegerTrait::<i8>::new(22_u8, true),
            IntegerTrait::<i8>::new(24_u8, true)
        ]
            .span(),
    );

    let a_scale = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(2000, false)].span(),
    );
    let a_zero_point = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),
    );
    let b_scale = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(2500, false)].span(),
    );
    let b_zero_point = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),
    );

    let y_scale = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(3000, false)].span(),
    );
    let y_zero_point = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),
    );

    let actual_output = a
        .qlinear_matmul(
            @a_scale, @a_zero_point, @b, @b_scale, @b_zero_point, @y_scale, @y_zero_point
        );
    let expected_output = TensorTrait::<i8>::new(
        shape: array![2, 3].span(),
        data: array![
            IntegerTrait::<i8>::new(3_u8, true),
            IntegerTrait::<i8>::new(4_u8, true),
            IntegerTrait::<i8>::new(4_u8, true),
            IntegerTrait::<i8>::new(8_u8, true),
            IntegerTrait::<i8>::new(9_u8, true),
            IntegerTrait::<i8>::new(10_u8, true)
        ]
            .span(),
    );

    assert((*actual_output.data[0]).into() == 3, '*result[0] == 3');
    assert((*actual_output.data[1]).into() == 4, '*result[1] == 4');
    assert((*actual_output.data[2]).into() == 4, '*result[2] == 4');
    assert((*actual_output.data[3]).into() == 8, '*result[3] == 8');
    assert((*actual_output.data[4]).into() == 9, '*result[4] == 9');
    assert((*actual_output.data[5]).into() == 10, '*result[5] == 10');

}


#[test]
#[available_gas(200000000000)]
fn qlinearmatmul_3D_test() {
    let a = TensorTrait::<i8>::new(
        shape: array![2, 2, 3].span(),
        data: array![
            IntegerTrait::<i8>::new(1_u8, true),
            IntegerTrait::<i8>::new(2_u8, true),
            IntegerTrait::<i8>::new(2_u8, true),
            IntegerTrait::<i8>::new(3_u8, true),
            IntegerTrait::<i8>::new(4_u8, true),
            IntegerTrait::<i8>::new(4_u8, true),
            IntegerTrait::<i8>::new(5_u8, true),
            IntegerTrait::<i8>::new(6_u8, true),
            IntegerTrait::<i8>::new(6_u8, true),
            IntegerTrait::<i8>::new(7_u8, true),
            IntegerTrait::<i8>::new(8_u8, true),
            IntegerTrait::<i8>::new(8_u8, true)
        ]
            .span(),
    );
    let b = TensorTrait::<i8>::new(
        shape: array![2, 3, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(2_u8, true),
            IntegerTrait::<i8>::new(4_u8, true),
            IntegerTrait::<i8>::new(6_u8, true),
            IntegerTrait::<i8>::new(8_u8, true),
            IntegerTrait::<i8>::new(10_u8, true),
            IntegerTrait::<i8>::new(12_u8, true),
            IntegerTrait::<i8>::new(2_u8, true),
            IntegerTrait::<i8>::new(4_u8, true),
            IntegerTrait::<i8>::new(6_u8, true),
            IntegerTrait::<i8>::new(8_u8, true),
            IntegerTrait::<i8>::new(10_u8, true),
            IntegerTrait::<i8>::new(12_u8, true)
        ]
            .span(),
    );

    let a_scale = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(20000, false)].span(),
    );
    let a_zero_point = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),
    );
    let b_scale = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(25000, false)].span(),
    );
    let b_zero_point = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),
    );

    let y_scale = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(30000, false)].span(),
    );
    let y_zero_point = TensorTrait::<FP16x16>::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),
    );

    let actual_output = a
        .qlinear_matmul(
            @a_scale, @a_zero_point, @b, @b_scale, @b_zero_point, @y_scale, @y_zero_point
        );
    let expected_output = TensorTrait::<i8>::new(
        shape: array![2, 2, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(8_u8, true),
            IntegerTrait::<i8>::new(11_u8, true),
            IntegerTrait::<i8>::new(17_u8, true),
            IntegerTrait::<i8>::new(23_u8, true),
            IntegerTrait::<i8>::new(26_u8, true),
            IntegerTrait::<i8>::new(35_u8, true),
            IntegerTrait::<i8>::new(36_u8, true),
            IntegerTrait::<i8>::new(47_u8, true)
        ]
            .span(),
    );
    assert((*actual_output.data[0]).into() == 8, '*result[0] == 8');
    assert((*actual_output.data[1]).into() == 11, '*result[1] == 11');
    assert((*actual_output.data[2]).into() == 17, '*result[2] == 17');
    assert((*actual_output.data[3]).into() == 23, '*result[3] == 23');
    assert((*actual_output.data[4]).into() == 26, '*result[4] == 26');
    assert((*actual_output.data[5]).into() == 35, '*result[5] == 35');
    assert((*actual_output.data[6]).into() == 36, '*result[6] == 36');
    assert((*actual_output.data[7]).into() == 47, '*result[7] == 47');

}

fn print_span(mut span: Span<i8>) {
    loop {
        match span.pop_front() {
            Option::Some(i) => {
                (*i.mag).print();
            },
            Option::None(_) => {
                break;
            }
        };
    };
}
