use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(98);
    data.append(89);
    data.append(-126);
    data.append(-68);
    data.append(31);
    data.append(7);
    data.append(-86);
    data.append(99);
    TensorTrait::new(shape.span(), data.span())
}
