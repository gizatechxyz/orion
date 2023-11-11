use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(52);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(118);
    data.append(122);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(176);
    data.append(97);
    data.append(234);
    data.append(0);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
