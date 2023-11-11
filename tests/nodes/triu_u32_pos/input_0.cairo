use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(87);
    data.append(183);
    data.append(123);
    data.append(66);
    data.append(19);
    data.append(149);
    data.append(68);
    data.append(125);
    data.append(0);
    data.append(159);
    data.append(126);
    data.append(205);
    data.append(123);
    data.append(17);
    data.append(4);
    data.append(174);
    data.append(92);
    data.append(233);
    data.append(181);
    data.append(26);
    TensorTrait::new(shape.span(), data.span())
}
