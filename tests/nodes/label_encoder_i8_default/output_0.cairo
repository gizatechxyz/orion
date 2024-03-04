use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(11);

    let mut data = ArrayTrait::new();
    data.append(11);
    data.append(22);
    data.append(99);
    data.append(99);
    data.append(55);
    data.append(66);
    data.append(11);
    data.append(22);
    data.append(99);
    data.append(77);
    data.append(99);
    TensorTrait::new(shape.span(), data.span())
}
