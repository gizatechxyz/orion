use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(20);
    data.append(0);
    data.append(0);
    data.append(130);
    data.append(92);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(191);
    data.append(0);
    data.append(0);
    data.append(175);
    data.append(177);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
