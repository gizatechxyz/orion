use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(-24);
    data.append(-16);
    data.append(-113);
    data.append(-11);
    data.append(-67);
    data.append(53);
    data.append(65);
    data.append(29);
    TensorTrait::new(shape.span(), data.span())
}
