use array::SpanTrait;
use traits::Into;
use array::ArrayTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::operators::tensor::core::{TensorTrait,Tensor, ExtraParams };
use orion::numbers::fixed_point::implementations::impl_16x16;

#[test]
#[available_gas(20000000)]
fn tanh_i32_test() {
    
    let mut sizes = ArrayTrait::new();
    sizes.append(5);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(2_u32, true));
    data.append(IntegerTrait::new(1_u32, true));
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    let result = tensor.tanh().data; 

    assert((*result.at(0).mag).into() == 63178, 'result[0] = 0.96403');
    assert(*result.at(0).sign == true, 'result[0] = true');
    assert((*result.at(1).mag).into() == 49911, 'result[1] = 0.76159');
    assert(*result.at(1).sign == true, 'result[1] = true');
    assert((*result.at(2).mag).into() == 0, 'result[2] = 0');
    assert(*result.at(3).sign == false, 'result[3] = false');
    assert((*result.at(3).mag).into() == 49911, 'result[3] = 0.76159');
    assert(*result.at(4).sign == false, 'result[4] = false');
    assert((*result.at(4).mag).into() == 63178, 'result[4] = 0.96403');
}

