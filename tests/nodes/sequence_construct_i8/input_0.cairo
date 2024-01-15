use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorSub};

fn input_0() -> Array<Tensor<i8>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(5);
    data.append(-2);
    data.append(-3);
    data.append(0);
    data.append(-3);
    data.append(5);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(5);
    data.append(2);
    data.append(-1);
    data.append(0);
    data.append(-5);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(3);
    data.append(-2);
    data.append(-1);
    data.append(-1);
    data.append(1);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 3, sign: false });
    data.append(i8 { mag: 1, sign: true });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 2, sign: true });
    data.append(i8 { mag: 3, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
