use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(-22);
    data.append(0);
    data.append(0);
    data.append(-51);
    data.append(13);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(-12);
    data.append(0);
    data.append(0);
    data.append(64);
    data.append(105);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
