use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorSub};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(110);
    data.append(-107);
    data.append(-71);
    data.append(113);
    data.append(26);
    data.append(-96);
    data.append(31);
    data.append(-118);
    TensorTrait::new(shape.span(), data.span())
}
