use array::SpanTrait;
use array::{ArrayTrait};
use orion::operators::tensor::implementations::impl_tensor_u32;
use orion::numbers::fixed_point::implementations::impl_16x16;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use traits::Into;

#[test]
#[available_gas(200000000)]
fn tanh_u32() {
let mut sizes = ArrayTrait::new();
    sizes.append(3);
    
    let mut arr = ArrayTrait::<u32>::new();
    arr.append(0_u32);
    arr.append(1_u32);
    arr.append(2_u32);

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<u32>::new(sizes.span(), arr.span(), extra);
   
    let result = tensor.tanh().data;

    assert(*result.at(0).mag.into() == 0, 'result[0] = 0');
    assert(*result.at(1).mag.into() == 49911, 'result[1] = 0.76159');
    assert(*result.at(2).mag.into() == 63178, 'result[2] = 0.96403');

}