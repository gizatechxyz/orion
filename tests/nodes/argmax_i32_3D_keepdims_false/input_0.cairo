use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(-121);
    data.append(-114);
    data.append(56);
    data.append(42);
    data.append(79);
    data.append(43);
    data.append(126);
    data.append(-64);
    TensorTrait::new(shape.span(), data.span())
}
