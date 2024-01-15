use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorSub};

fn output_0() -> Array<Tensor<i8>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(0);

    let mut data = ArrayTrait::new();

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
