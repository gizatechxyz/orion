use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorSub};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(-10);
    data.append(20);
    data.append(8);
    data.append(-10);
    data.append(20);
    data.append(-10);
    data.append(1);
    data.append(-10);
    TensorTrait::new(shape.span(), data.span())
}
