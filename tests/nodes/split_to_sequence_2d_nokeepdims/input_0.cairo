use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(8);

    let mut data = ArrayTrait::new();
    data.append(32);
    data.append(67);
    data.append(110);
    data.append(16);
    data.append(154);
    data.append(139);
    data.append(43);
    data.append(0);
    data.append(104);
    data.append(246);
    data.append(70);
    data.append(120);
    data.append(221);
    data.append(191);
    data.append(140);
    data.append(118);
    TensorTrait::new(shape.span(), data.span())
}
