use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(18);

    let mut data = ArrayTrait::new();
    data.append(4);
    data.append(200);
    data.append(228);
    data.append(237);
    data.append(202);
    data.append(44);
    data.append(18);
    data.append(52);
    data.append(219);
    data.append(21);
    data.append(92);
    data.append(17);
    data.append(88);
    data.append(66);
    data.append(54);
    data.append(201);
    data.append(244);
    data.append(252);
    TensorTrait::new(shape.span(), data.span())
}
