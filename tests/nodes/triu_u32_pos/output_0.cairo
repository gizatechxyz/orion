use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(0);
    data.append(123);
    data.append(66);
    data.append(19);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(159);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(4);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
