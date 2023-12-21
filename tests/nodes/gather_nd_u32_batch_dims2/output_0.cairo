use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(13);
    data.append(25);
    data.append(37);
    data.append(51);
    data.append(64);
    data.append(75);
    data.append(85);
    data.append(96);
    TensorTrait::new(shape.span(), data.span())
}
