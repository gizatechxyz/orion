use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(124);
    data.append(25);
    data.append(157);
    data.append(20);
    data.append(170);
    data.append(58);
    data.append(130);
    data.append(92);
    data.append(31);
    data.append(125);
    data.append(67);
    data.append(207);
    data.append(191);
    data.append(46);
    data.append(237);
    data.append(175);
    data.append(177);
    data.append(237);
    TensorTrait::new(shape.span(), data.span())
}
