use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(-56);
    data.append(-29);
    data.append(-39);
    data.append(0);
    data.append(0);
    data.append(-40);
    data.append(66);
    data.append(-38);
    data.append(70);
    data.append(0);
    data.append(83);
    data.append(-81);
    data.append(57);
    data.append(-48);
    data.append(-71);
    data.append(109);
    data.append(-14);
    data.append(25);
    data.append(-113);
    data.append(88);
    TensorTrait::new(shape.span(), data.span())
}
