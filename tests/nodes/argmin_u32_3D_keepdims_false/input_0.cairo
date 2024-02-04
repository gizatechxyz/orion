use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(133);
    data.append(237);
    data.append(19);
    data.append(140);
    data.append(162);
    data.append(214);
    data.append(247);
    data.append(234);
    TensorTrait::new(shape.span(), data.span())
}
