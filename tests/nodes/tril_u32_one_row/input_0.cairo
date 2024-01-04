use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(68);
    data.append(176);
    data.append(238);
    data.append(164);
    data.append(157);
    data.append(211);
    data.append(97);
    data.append(132);
    data.append(224);
    data.append(245);
    data.append(118);
    data.append(25);
    data.append(196);
    data.append(43);
    data.append(124);
    TensorTrait::new(shape.span(), data.span())
}
