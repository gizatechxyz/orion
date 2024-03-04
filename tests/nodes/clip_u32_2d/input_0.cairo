use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorSub};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(102);
    data.append(105);
    data.append(111);
    data.append(39);
    data.append(165);
    data.append(3);
    data.append(165);
    data.append(196);
    TensorTrait::new(shape.span(), data.span())
}
