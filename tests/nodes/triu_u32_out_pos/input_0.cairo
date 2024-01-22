use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(93);
    data.append(140);
    data.append(31);
    data.append(242);
    data.append(44);
    data.append(11);
    data.append(240);
    data.append(166);
    data.append(214);
    data.append(85);
    data.append(168);
    data.append(254);
    data.append(114);
    data.append(206);
    data.append(145);
    data.append(171);
    data.append(195);
    data.append(189);
    data.append(208);
    data.append(50);
    TensorTrait::new(shape.span(), data.span())
}
