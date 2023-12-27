use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(-120);
    data.append(122);
    data.append(-75);
    data.append(123);
    data.append(-48);
    data.append(-96);
    data.append(-113);
    data.append(59);
    TensorTrait::new(shape.span(), data.span())
}
