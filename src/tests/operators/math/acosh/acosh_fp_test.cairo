use array::SpanTrait;
use array::{ArrayTrait};
use traits::Into;
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::impl_8x23;

#[test]
#[available_gas(2000000000)]
fn acosh_fp_test() {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut arr = ArrayTrait::<FixedType>::new();
    
    arr.append(FixedTrait::new_unscaled(1, false));
    arr.append(FixedTrait::new_unscaled(2, false));
    arr.append(FixedTrait::new_unscaled(3, false));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), arr.span(), Option::Some(extra));

    let result = tensor.acosh().data;

    assert((*result.at(0).mag) == 0, 'result[0] = 0');
    assert((*result.at(1).mag) == 11047444, 'result[1] = 1.31696');
    assert((*result.at(2).mag) == 14787433, 'result[2] = 1.76275');

}

#[test]
#[available_gas(2000000000)]
#[should_panic]
fn acosh_neg_example() {
    
    let mut sizes = ArrayTrait::new();
    sizes.append(1);

    let mut arr = ArrayTrait::<FixedType>::new();
    arr.append(FixedTrait::new_unscaled(1, true));
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), arr.span(), Option::Some(extra));
    // should panic with a negative value
    tensor.acosh();

}

#[test]
#[available_gas(2000000000)]
#[should_panic]
fn acosh_zero_example() {
    
    let mut sizes = ArrayTrait::new();
    sizes.append(1);

    let mut arr = ArrayTrait::<FixedType>::new();
    arr.append(FixedTrait::new_unscaled(0, false));
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), arr.span(), Option::Some(extra));
    // should panic with a zero value
    tensor.acosh();

}