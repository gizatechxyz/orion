use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorSub};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(4);
    data.append(4);
    data.append(3);
    data.append(3);
    data.append(6);
    data.append(7);
    data.append(7);
    data.append(6);
    data.append(9);
    data.append(10);
    data.append(9);
    data.append(9);
    data.append(13);
    data.append(12);
    data.append(12);
    data.append(13);
    data.append(15);
    data.append(16);
    data.append(15);
    data.append(16);
    data.append(19);
    data.append(19);
    data.append(18);
    data.append(18);
    data.append(21);
    data.append(22);
    data.append(21);
    data.append(21);
    TensorTrait::new(shape.span(), data.span())
}
