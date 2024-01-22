use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(88);
    data.append(-82);
    data.append(122);
    data.append(55);
    data.append(115);
    data.append(-98);
    data.append(0);
    data.append(-40);
    data.append(-106);
    data.append(-61);
    data.append(-108);
    data.append(-84);
    data.append(-86);
    data.append(48);
    data.append(-10);
    data.append(0);
    data.append(36);
    data.append(114);
    TensorTrait::new(shape.span(), data.span())
}
