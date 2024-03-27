use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(113);
    data.append(207);
    data.append(34);
    data.append(249);
    data.append(186);
    data.append(56);
    data.append(153);
    data.append(92);
    TensorTrait::new(shape.span(), data.span())
}
