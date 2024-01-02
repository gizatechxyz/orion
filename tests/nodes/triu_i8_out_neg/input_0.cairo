use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(111);
    data.append(-62);
    data.append(-62);
    data.append(-51);
    data.append(-68);
    data.append(-61);
    data.append(102);
    data.append(-62);
    data.append(84);
    data.append(-41);
    data.append(80);
    data.append(-26);
    data.append(-115);
    data.append(95);
    data.append(46);
    data.append(119);
    data.append(-127);
    data.append(-51);
    data.append(48);
    data.append(14);
    TensorTrait::new(shape.span(), data.span())
}
