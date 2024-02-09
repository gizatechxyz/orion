use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(25);
    data.append(65);
    data.append(12);
    data.append(81);
    data.append(41);
    data.append(50);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(9);
    data.append(10);
    data.append(11);
    TensorTrait::new(shape.span(), data.span())
}
