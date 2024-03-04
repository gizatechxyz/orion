use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(83);
    data.append(103);
    data.append(-56);
    data.append(0);
    data.append(99);
    data.append(23);
    data.append(0);
    data.append(0);
    data.append(126);
    data.append(58);
    data.append(-92);
    data.append(-73);
    data.append(0);
    data.append(99);
    data.append(80);
    data.append(0);
    data.append(0);
    data.append(108);
    TensorTrait::new(shape.span(), data.span())
}
