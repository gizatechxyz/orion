use array::SpanTrait;
use traits::Into;
use array::ArrayTrait;
use debug::PrintTrait;


use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams };
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::tests::operators::tensor::helpers::i32_tensor_1x3_helper;
use orion::numbers::fixed_point::implementations::impl_16x16;

#[test]
#[available_gas(20000000)]
fn cosh_test() {
    
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
    
    
    let result = tensor.cosh().data; 

    let result1 = *result.at(0);
    result1.print();
    let result2 = *result.at(1);
    result2.print();
    let result3 = *result.at(2);
    result3.print();
    let result4 = *result.at(3);
    result4.print();
    let result5 = *result.at(4);
    result5.print();

    assert((*result.at(0).mag).into() == 246550, 'result[0] = 3.7622');
    assert(*result.at(0).sign == false, 'result[0] = false');
    assert((*result.at(1).mag).into() == 101125, 'result[1] = 1.5431');
    assert(*result.at(1).sign == false, 'result[1] = false');
    assert((*result.at(2).mag).into() == 65536, 'result[2] = 1');
    assert(*result.at(3).sign == false, 'result[3] = false');
    assert((*result.at(3).mag).into() == 101125, 'result[3] = 1.5431');
    assert(*result.at(4).sign == false, 'result[4] = false');
    assert((*result.at(4).mag).into() == 246550, 'result[4] = 3.7622');
}

