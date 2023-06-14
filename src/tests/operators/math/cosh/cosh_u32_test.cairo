use array::SpanTrait;
use array::{ArrayTrait};
use orion::operators::tensor::implementations::impl_tensor_u32;
use orion::numbers::fixed_point::implementations::impl_16x16;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use traits::Into;


#[test]
#[available_gas(200000000)]
fn cosh_u32() {
let mut sizes = ArrayTrait::new();
    sizes.append(3);
    
    let mut arr = ArrayTrait::<u32>::new();
    arr.append(0_u32);
    arr.append(1_u32);
    arr.append(2_u32);

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<u32>::new(sizes.span(), arr.span(), extra);
   
    let result = tensor.cosh().data;

    assert(*result.at(0).mag.into() == 65536, 'result[0] = 1');
    assert(*result.at(1).mag.into() == 101125, 'result[1] = 1.5431');
    assert(*result.at(2).mag.into() == 246550, 'result[2] = 3.7622');

}