use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorDiv};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(2);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(2);
    data.append(2);
    data.append(1);
    data.append(1);
    data.append(2);
    data.append(1);
    data.append(2);
    data.append(2);
    data.append(2);
    data.append(2);
    data.append(2);
    data.append(2);
    data.append(1);
    data.append(1);
    data.append(2);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(2);
    data.append(1);
    data.append(1);
    data.append(2);
    TensorTrait::new(shape.span(), data.span())
}
