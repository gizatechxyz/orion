use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(30);
    data.append(33);
    data.append(36);
    data.append(39);
    data.append(42);
    data.append(45);
    data.append(48);
    data.append(51);
    data.append(54);
    TensorTrait::new(shape.span(), data.span())
}
