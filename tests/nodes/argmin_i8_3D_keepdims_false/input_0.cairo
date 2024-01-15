use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(-7);
    data.append(-74);
    data.append(93);
    data.append(-43);
    data.append(77);
    data.append(-8);
    data.append(39);
    data.append(-121);
    TensorTrait::new(shape.span(), data.span())
}
