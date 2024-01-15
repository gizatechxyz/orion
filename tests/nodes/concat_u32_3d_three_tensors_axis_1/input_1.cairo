use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(27);
    data.append(28);
    data.append(29);
    data.append(30);
    data.append(31);
    data.append(32);
    data.append(33);
    data.append(34);
    data.append(35);
    data.append(36);
    data.append(37);
    data.append(38);
    data.append(39);
    data.append(40);
    data.append(41);
    data.append(42);
    data.append(43);
    data.append(44);
    data.append(45);
    data.append(46);
    data.append(47);
    data.append(48);
    data.append(49);
    data.append(50);
    data.append(51);
    data.append(52);
    data.append(53);
    TensorTrait::new(shape.span(), data.span())
}
