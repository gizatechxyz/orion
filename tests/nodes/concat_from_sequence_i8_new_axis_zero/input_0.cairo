use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Array<Tensor<i8>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(4);
    data.append(0);
    data.append(-6);
    data.append(-1);
    data.append(4);
    data.append(-6);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-5);
    data.append(-2);
    data.append(4);
    data.append(1);
    data.append(-2);
    data.append(-5);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    data.append(2);
    data.append(-3);
    data.append(5);
    data.append(2);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-2);
    data.append(-5);
    data.append(-2);
    data.append(-2);
    data.append(-1);
    data.append(1);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(2);
    data.append(-2);
    data.append(-3);
    data.append(2);
    data.append(3);
    data.append(0);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
