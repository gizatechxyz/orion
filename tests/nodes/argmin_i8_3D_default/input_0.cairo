use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(104);
    data.append(91);
    data.append(-75);
    data.append(51);
    data.append(-57);
    data.append(-22);
    data.append(-13);
    data.append(72);
    TensorTrait::new(shape.span(), data.span())
}
