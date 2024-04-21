use orion::operators::tensor::{TensorTrait, Tensor};

/// Cf: SequenceTrait::sequence_length docstring
fn sequence_length<T, impl TDrop: Drop<T>>(self: Array<Tensor<T>>) -> Tensor<u32> {
    let mut shape: Array<usize> = array![];
    let mut result: Array<usize> = array![];
    result.append(self.len());

    Tensor::<u32> { shape: shape.span(), data: result.span(), }
}
