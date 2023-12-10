use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor};


/// Cf: TensorTrait::sequence_empty docstring
fn sequence_empty<T, impl TTensorTrait: TensorTrait<T>, impl TDrop: Drop<T>>() -> Array<Tensor<T>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(0);

    let mut data = ArrayTrait::new();
    let tensor = TensorTrait::new(shape.span(), data.span());

    sequence.append(tensor);

    sequence
}
