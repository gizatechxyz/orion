use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    data.append(5);
    data.append(6);
    data.append(9);
    data.append(10);
    data.append(13);
    data.append(14);
    data.append(17);
    data.append(18);
    data.append(21);
    data.append(22);
    data.append(25);
    data.append(26);
    data.append(29);
    data.append(30);
    data.append(33);
    data.append(34);
    data.append(37);
    data.append(38);
    data.append(41);
    data.append(42);
    data.append(45);
    data.append(46);
    TensorTrait::new(shape.span(), data.span())
}
