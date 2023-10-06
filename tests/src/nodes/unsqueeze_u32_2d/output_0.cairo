use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(2);
    shape.append(4);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(118);
    data.append(15);
    data.append(98);
    data.append(189);
    data.append(191);
    data.append(31);
    data.append(239);
    data.append(238);
    TensorTrait::new(shape.span(), data.span())
}