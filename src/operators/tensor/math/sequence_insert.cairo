use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::i32;


/// Cf: TensorTrait::sequence_insert docstring
fn sequence_insert<T, impl TDrop: Drop<T>>(self: Array<Tensor<T>>, tensor: @Tensor<T>, position: @Tensor<i32>) -> Array<Tensor<T>> {
    self
}
