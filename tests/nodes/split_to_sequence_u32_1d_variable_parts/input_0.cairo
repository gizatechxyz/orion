use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(18);

    let mut data = ArrayTrait::new();
    data.append(62);
    data.append(252);
    data.append(185);
    data.append(102);
    data.append(157);
    data.append(185);
    data.append(33);
    data.append(77);
    data.append(96);
    data.append(222);
    data.append(216);
    data.append(238);
    data.append(94);
    data.append(17);
    data.append(186);
    data.append(228);
    data.append(118);
    data.append(254);
    TensorTrait::new(shape.span(), data.span())
}
