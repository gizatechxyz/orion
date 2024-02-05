use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(37);
    data.append(213);
    data.append(154);
    data.append(116);
    data.append(39);
    data.append(6);
    data.append(241);
    data.append(106);
    data.append(87);
    data.append(165);
    data.append(25);
    data.append(118);
    data.append(46);
    data.append(175);
    data.append(186);
    data.append(105);
    data.append(118);
    data.append(149);
    TensorTrait::new(shape.span(), data.span())
}
