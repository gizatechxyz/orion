use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(-122);
    data.append(-111);
    data.append(98);
    data.append(108);
    data.append(-127);
    data.append(-126);
    data.append(-38);
    data.append(93);
    data.append(-64);
    data.append(-68);
    data.append(9);
    data.append(79);
    data.append(83);
    data.append(17);
    data.append(62);
    data.append(56);
    data.append(123);
    data.append(97);
    data.append(102);
    data.append(87);
    TensorTrait::new(shape.span(), data.span())
}
