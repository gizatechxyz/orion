use orion::operators::tensor::{TensorTrait, Tensor};

/// Cf: SequenceTrait::sequence_empty docstring
fn sequence_empty<T, impl TTensorTrait: TensorTrait<T>, impl TDrop: Drop<T>>() -> Array<Tensor<T>> {
    let mut sequence = array![];

    let mut shape: Array<usize> = array![];
    shape.append(0);

    let mut data: Array<T> = array![];
    let tensor = TensorTrait::new(shape.span(), data.span());

    sequence.append(tensor);

    sequence
}
