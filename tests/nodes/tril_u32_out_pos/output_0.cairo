use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(145);
    data.append(194);
    data.append(234);
    data.append(81);
    data.append(44);
    data.append(119);
    data.append(171);
    data.append(202);
    data.append(17);
    data.append(253);
    data.append(114);
    data.append(238);
    data.append(36);
    data.append(144);
    data.append(165);
    data.append(81);
    data.append(249);
    data.append(43);
    data.append(223);
    data.append(247);
    TensorTrait::new(shape.span(), data.span())
}
