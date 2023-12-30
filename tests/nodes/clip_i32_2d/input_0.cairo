use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorSub};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(-72);
    data.append(41);
    data.append(8);
    data.append(-66);
    data.append(62);
    data.append(-57);
    data.append(1);
    data.append(-34);
    TensorTrait::new(shape.span(), data.span())
}
