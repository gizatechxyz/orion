use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(207);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(6);
    data.append(88);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(163);
    data.append(160);
    data.append(133);
    data.append(0);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
