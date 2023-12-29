use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorSub};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(123);
    data.append(12);
    data.append(-91);
    data.append(122);
    data.append(66);
    data.append(42);
    data.append(12);
    data.append(18);
    TensorTrait::new(shape.span(), data.span())
}
