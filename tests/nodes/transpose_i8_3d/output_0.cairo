use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorSub};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(68);
    data.append(95);
    data.append(-71);
    data.append(-9);
    data.append(50);
    data.append(-101);
    data.append(117);
    data.append(-14);
    TensorTrait::new(shape.span(), data.span())
}
