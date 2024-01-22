use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(21);
    data.append(206);
    data.append(245);
    data.append(180);
    data.append(229);
    data.append(107);
    data.append(167);
    data.append(193);
    TensorTrait::new(shape.span(), data.span())
}
