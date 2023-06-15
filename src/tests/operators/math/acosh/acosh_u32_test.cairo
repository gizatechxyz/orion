use array::SpanTrait;
use traits::Into;
use array::ArrayTrait;

use orion::operators::tensor::implementations::impl_tensor_u32;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams };
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::numbers::fixed_point::implementations::impl_16x16;

use debug::PrintTrait;

#[test]
#[available_gas(2000000000)]
fn acosh_u32_test() {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    
    let mut arr = ArrayTrait::<u32>::new();
    arr.append(1_u32);
    arr.append(2_u32);
    arr.append(3_u32);
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<u32>::new(sizes.span(), arr.span(), extra);

    let result = tensor.acosh().data;
    
    assert((*result.at(0).mag.into()) == 0, 'result[0] = 0');
    assert((*result.at(1).mag.into()) == 86255, 'result[1] = 1.31696');
    assert((*result.at(2).mag.into()) == 115516, 'result[2] = 1.76275');

}