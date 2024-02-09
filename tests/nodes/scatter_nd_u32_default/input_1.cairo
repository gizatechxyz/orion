use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(41);
    data.append(6);
    data.append(63);
    data.append(34);
    data.append(67);
    data.append(10);
    TensorTrait::new(shape.span(), data.span())
}
