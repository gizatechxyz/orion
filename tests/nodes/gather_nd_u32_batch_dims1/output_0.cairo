use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(12);
    data.append(13);
    data.append(14);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(36);
    data.append(37);
    data.append(38);
    data.append(51);
    data.append(52);
    data.append(53);
    data.append(39);
    data.append(40);
    data.append(41);
    data.append(72);
    data.append(73);
    data.append(74);
    data.append(72);
    data.append(73);
    data.append(74);
    data.append(87);
    data.append(88);
    data.append(89);
    TensorTrait::new(shape.span(), data.span())
}
