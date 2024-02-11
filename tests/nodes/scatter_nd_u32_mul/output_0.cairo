use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(42);
    data.append(144);
    data.append(225);
    data.append(16);
    data.append(70);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(9);
    data.append(10);
    data.append(11);
    TensorTrait::new(shape.span(), data.span())
}
