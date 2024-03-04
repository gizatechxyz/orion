use orion::operators::tensor::{TensorTrait, Tensor};

/// Cf: SequenceTrait::sequence_construct docstring
fn sequence_construct<T, impl TDrop: Drop<T>>(tensors: Array<Tensor<T>>) -> Array<Tensor<T>> {
    assert(tensors.len() >= 1, 'Input tensors must be >= 1');

    tensors
}
