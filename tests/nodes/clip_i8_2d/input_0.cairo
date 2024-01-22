use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorSub};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(-29);
    data.append(82);
    data.append(121);
    data.append(-89);
    data.append(-87);
    data.append(69);
    data.append(71);
    data.append(-25);
    TensorTrait::new(shape.span(), data.span())
}
