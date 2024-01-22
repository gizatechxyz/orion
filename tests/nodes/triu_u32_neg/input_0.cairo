use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(13);
    data.append(227);
    data.append(211);
    data.append(51);
    data.append(240);
    data.append(178);
    data.append(74);
    data.append(169);
    data.append(148);
    data.append(211);
    data.append(154);
    data.append(235);
    data.append(49);
    data.append(208);
    data.append(247);
    data.append(159);
    data.append(153);
    data.append(46);
    data.append(159);
    data.append(204);
    TensorTrait::new(shape.span(), data.span())
}
