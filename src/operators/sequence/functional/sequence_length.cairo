use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor};


/// Cf: SequenceTrait::sequence_length docstring
fn sequence_length<T, impl TDrop: Drop<T>>(self: Array<Tensor<T>>) -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    let mut result = ArrayTrait::new();
    result.append(self.len());

    Tensor::<u32> { shape: shape.span(), data: result.span(), }
}
