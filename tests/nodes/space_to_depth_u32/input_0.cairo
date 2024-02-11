use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(4294967295);
    data.append(4294967295);
    data.append(0);
    data.append(4294967294);
    data.append(2);
    data.append(4294967295);
    data.append(4294967294);
    data.append(4294967294);
    data.append(4294967295);
    data.append(4294967295);
    data.append(2);
    data.append(2);
    data.append(1);
    data.append(4294967293);
    data.append(4294967294);
    data.append(2);
    TensorTrait::new(shape.span(), data.span())
}
