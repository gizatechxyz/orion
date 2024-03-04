use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorSub};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(82);
    data.append(-24);
    data.append(-112);
    data.append(-21);
    data.append(-51);
    data.append(56);
    data.append(13);
    data.append(9);
    TensorTrait::new(shape.span(), data.span())
}
