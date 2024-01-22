use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    data.append(6);
    data.append(7);
    data.append(11);
    data.append(12);
    data.append(16);
    data.append(17);
    data.append(21);
    data.append(22);
    data.append(26);
    data.append(27);
    data.append(31);
    data.append(32);
    data.append(36);
    data.append(37);
    data.append(41);
    data.append(42);
    data.append(46);
    data.append(47);
    data.append(51);
    data.append(52);
    data.append(56);
    data.append(57);
    TensorTrait::new(shape.span(), data.span())
}
