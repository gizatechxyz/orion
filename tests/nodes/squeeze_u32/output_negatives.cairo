use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn negatives() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(1);
    TensorTrait::new(shape.span(), data.span())
}
