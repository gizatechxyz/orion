use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorSub};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(-70);
    data.append(71);
    data.append(121);
    data.append(106);
    data.append(-89);
    data.append(11);
    data.append(-69);
    data.append(-78);
    TensorTrait::new(shape.span(), data.span())
}
