use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use debug::PrintTrait;


#[test]
#[available_gas(200000000000)]
fn transpose_test_shape() {
    let tensor = TensorTrait::<u32>::new(
        shape: array![4, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    let result = tensor.transpose(axes: array![1, 0].span());
    assert(result.shape == array![2, 4].span(), 'wrong dim');
}

#[test]
#[available_gas(200000000000)]
fn transpose_test_values() {
    let tensor = TensorTrait::<u32>::new(
        shape: array![4, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    let result = tensor.transpose(axes: array![1, 0].span());
    assert(result.data == array![0, 2, 4, 6, 1, 3, 5, 7].span(), 'wrong data');
}


#[test]
#[available_gas(200000000000)]
fn transpose_test_3D() {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    let result = tensor.transpose(axes: array![1, 2, 0].span());

    assert(result.shape == array![2, 2, 2].span(), 'wrong shape');
    assert(result.data == array![0, 4, 1, 5, 2, 6, 3, 7].span(), 'wrong data');
}

