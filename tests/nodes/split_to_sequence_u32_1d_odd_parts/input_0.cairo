use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(18);

    let mut data = ArrayTrait::new();
    data.append(30);
    data.append(31);
    data.append(133);
    data.append(227);
    data.append(179);
    data.append(119);
    data.append(10);
    data.append(68);
    data.append(116);
    data.append(137);
    data.append(12);
    data.append(69);
    data.append(28);
    data.append(166);
    data.append(93);
    data.append(65);
    data.append(169);
    data.append(229);
    TensorTrait::new(shape.span(), data.span())
}
