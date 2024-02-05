use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(18);

    let mut data = ArrayTrait::new();
    data.append(159);
    data.append(9);
    data.append(19);
    data.append(39);
    data.append(184);
    data.append(232);
    data.append(195);
    data.append(86);
    data.append(161);
    data.append(231);
    data.append(154);
    data.append(193);
    data.append(97);
    data.append(18);
    data.append(124);
    data.append(138);
    data.append(137);
    data.append(7);
    TensorTrait::new(shape.span(), data.span())
}
