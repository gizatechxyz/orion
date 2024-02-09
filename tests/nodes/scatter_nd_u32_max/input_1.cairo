use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(23);
    data.append(0);
    data.append(0);
    data.append(16);
    data.append(12);
    data.append(35);
    TensorTrait::new(shape.span(), data.span())
}
