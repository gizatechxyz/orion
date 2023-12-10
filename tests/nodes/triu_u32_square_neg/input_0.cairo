use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(38);
    data.append(103);
    data.append(89);
    data.append(80);
    data.append(35);
    data.append(166);
    data.append(47);
    data.append(229);
    data.append(247);
    data.append(77);
    data.append(3);
    data.append(229);
    data.append(236);
    data.append(225);
    data.append(89);
    data.append(27);
    data.append(43);
    data.append(253);
    TensorTrait::new(shape.span(), data.span())
}
