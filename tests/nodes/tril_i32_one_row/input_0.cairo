use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(48);
    data.append(-95);
    data.append(-5);
    data.append(-79);
    data.append(-24);
    data.append(69);
    data.append(-50);
    data.append(34);
    data.append(-91);
    data.append(24);
    data.append(-13);
    data.append(-117);
    data.append(107);
    data.append(102);
    data.append(77);
    TensorTrait::new(shape.span(), data.span())
}
