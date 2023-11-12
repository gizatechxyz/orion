use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use debug::PrintTrait;


#[test]
#[available_gas(200000000000)]
fn constant_of_shape_test() {
    let tensor = TensorTrait::<u32>::constant_of_shape(shape: array![4, 2].span(), value: 20);

    assert(tensor.shape == array![4, 2].span(), 'wrong dim');
    assert(tensor.data == array![20, 20, 20, 20, 20, 20, 20, 20].span(), 'wrong values');
}
