use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};
use orion::numbers::{NumberTrait, IntegerTrait};
use orion::numbers::{i8, i32};

#[test]
#[available_gas(200000000000)]
fn qlinear_concat_test() {
    let tensor1 = TensorTrait::<
        i8
    >::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(10_u8, false),
            IntegerTrait::<i8>::new(20_u8, false),
            IntegerTrait::<i8>::new(30_u8, false),
            IntegerTrait::<i8>::new(40_u8, false),
        ]
            .span(),
    );
    let tensor2 = TensorTrait::<
        i8
    >::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(20_u8, false),
            IntegerTrait::<i8>::new(40_u8, false),
            IntegerTrait::<i8>::new(60_u8, false),
            IntegerTrait::<i8>::new(80_u8, false),
        ]
            .span(),
    );

    let tensors = array![tensor1, tensor2].span();

    let tensor1_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(20000, false)].span(),);
    let tensor2_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(25000, false)].span(),);

    let scales = array![tensor1_scale, tensor2_scale].span();

    let tensor1_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),); 
    let tensor2_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);

    let zero_points = array![tensor1_zero_point, tensor2_zero_point].span();

    let y_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),);

    let y_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);

    let actual_output = TensorTrait::qlinear_concat(tensors, scales, zero_points, @y_scale, @y_zero_point, 0);
    
    assert((*actual_output.data[0]).into() == 3, '*result[0] == 3');
    assert((*actual_output.data[1]).into() == 6, '*result[1] == 6');
    assert((*actual_output.data[2]).into() == 9, '*result[2] == 9');
    assert((*actual_output.data[3]).into() == 12, '*result[3] == 12');
    assert((*actual_output.data[4]).into() == 7, '*result[4] == 8');
    assert((*actual_output.data[5]).into() == 15, '*result[5] == 15');
    assert((*actual_output.data[6]).into() == 24, '*result[6] == 24');
    assert((*actual_output.data[7]).into() == 40, '*result[7] == 40');
}
 
 

#[test]
#[available_gas(200000000000)]
fn qlinear_concat_test_shape() {
    let tensor1 = TensorTrait::<
        i8
    >::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(2_u8, false),
            IntegerTrait::<i8>::new(2_u8, false),
            IntegerTrait::<i8>::new(2_u8, false),
            IntegerTrait::<i8>::new(2_u8, false),
        ]
            .span(),
    );
    let tensor2 = TensorTrait::<
        i8
    >::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(8_u8, false),
            IntegerTrait::<i8>::new(8_u8, false),
            IntegerTrait::<i8>::new(8_u8, false),
            IntegerTrait::<i8>::new(8_u8, false),
        ]
            .span(),
    );
    let tensor3 = TensorTrait::<
        i8
    >::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(10_u8, false),
            IntegerTrait::<i8>::new(10_u8, false),
            IntegerTrait::<i8>::new(10_u8, false),
            IntegerTrait::<i8>::new(10_u8, false),
        ]
            .span(),
    );


    let tensors = array![tensor1, tensor2, tensor3].span();

    let tensor1_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),);
    let tensor2_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(262144, false)].span(),);
    let tensor3_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),);

    let scales = array![tensor1_scale, tensor2_scale, tensor3_scale].span();

    let tensor1_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),); 
    let tensor2_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),); 
    let tensor3_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),);

    let zero_points = array![tensor1_zero_point, tensor2_zero_point, tensor3_zero_point].span();

    let y_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(30000, false)].span(),);

    let y_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);

    let actual_output = TensorTrait::qlinear_concat(tensors, scales, zero_points, @y_scale, @y_zero_point, 0);

    assert((*actual_output.shape[0]).into() == 6, '*result.shape[0] == 6');
    assert((*actual_output.shape[1]).into() == 2, '*result.shape[1] == 2');

}
 


#[test]
#[available_gas(200000000000)]
fn qlinear_concat_example_doc() {
    let tensor1 = TensorTrait::<
        i8
    >::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(5_u8, false),
            IntegerTrait::<i8>::new(5_u8, false),
            IntegerTrait::<i8>::new(5_u8, false),
            IntegerTrait::<i8>::new(5_u8, false),
        ]
            .span(),
    );
    let tensor2 = TensorTrait::<
        i8
    >::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(1_u8, false),
            IntegerTrait::<i8>::new(1_u8, false),
            IntegerTrait::<i8>::new(1_u8, false),
            IntegerTrait::<i8>::new(1_u8, false),
        ]
            .span(),
    );

    let tensors = array![tensor1, tensor2].span();

    let tensor1_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),);
    let tensor2_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(262144, false)].span(),);

    let scales = array![tensor1_scale, tensor2_scale].span();

    let tensor1_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(327680, false)].span(),); 
    let tensor2_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);

    let zero_points = array![tensor1_zero_point, tensor2_zero_point].span();

    let y_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(262144, false)].span(),);

    let y_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),);

    let actual_output = TensorTrait::qlinear_concat(tensors, scales, zero_points, @y_scale, @y_zero_point, 0);
    
    assert((*actual_output.data[0]).into() == 1, '*result[0] == 1');
    assert((*actual_output.data[1]).into() == 1, '*result[1] == 1');
    assert((*actual_output.data[2]).into() == 1, '*result[2] == 1');
    assert((*actual_output.data[3]).into() == 1, '*result[3] == 1');
    assert((*actual_output.data[4]).into() == 2, '*result[4] == 2');
    assert((*actual_output.data[5]).into() == 2, '*result[5] == 2');
    assert((*actual_output.data[6]).into() == 2, '*result[4] == 2');
    assert((*actual_output.data[7]).into() == 2, '*result[5] == 2');
}