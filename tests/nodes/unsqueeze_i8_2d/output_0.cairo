use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorSub};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(2);
    shape.append(4);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(6);
    data.append(0);
    data.append(-30);
    data.append(99);
    data.append(-88);
    data.append(78);
    data.append(-59);
    data.append(76);
    TensorTrait::new(shape.span(), data.span())
}
