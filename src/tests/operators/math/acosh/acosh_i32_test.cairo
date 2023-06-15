use array::SpanTrait;
use traits::Into;
use array::ArrayTrait;

use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams };
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::tests::operators::tensor::helpers::i32_tensor_1x3_helper;
use orion::numbers::fixed_point::implementations::impl_16x16;

#[test]
#[available_gas(2000000000)]
fn acosh_i32_test() {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    data.append(IntegerTrait::new(3_u32, false));   
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    let result = tensor.acosh().data;
    
    assert((*result.at(0).mag.into()) == 0, 'result[0] = 0');
    assert((*result.at(1).mag.into()) == 86255, 'result[1] = 1.31696');
    assert((*result.at(2).mag.into()) == 115516, 'result[2] = 1.76275');

}

#[test]
#[available_gas(2000000000)]
#[should_panic]
fn acosh_neg_example() {
    
    let mut sizes = ArrayTrait::new();
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(1_u32, true));
 
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);
    // should panic with a negative value
    tensor.acosh();

}

#[test]
#[available_gas(2000000000)]
#[should_panic]
fn acosh_zero_example() {
    
    let mut sizes = ArrayTrait::new();
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
 
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);
    // should panic with a negative value
    tensor.acosh();

}