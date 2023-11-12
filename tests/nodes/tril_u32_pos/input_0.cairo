use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(83);
    data.append(229);
    data.append(59);
    data.append(240);
    data.append(114);
    data.append(209);
    data.append(118);
    data.append(37);
    data.append(203);
    data.append(215);
    data.append(49);
    data.append(166);
    data.append(119);
    data.append(199);
    data.append(190);
    data.append(187);
    data.append(3);
    data.append(24);
    data.append(217);
    data.append(121);
    TensorTrait::new(shape.span(), data.span())
}
