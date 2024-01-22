use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(95);
    data.append(53);
    data.append(-5);
    data.append(17);
    data.append(-86);
    data.append(-11);
    data.append(-95);
    data.append(104);
    data.append(12);
    data.append(-5);
    data.append(64);
    data.append(53);
    data.append(-52);
    data.append(48);
    data.append(-111);
    data.append(-13);
    data.append(63);
    data.append(-112);
    data.append(31);
    data.append(-65);
    TensorTrait::new(shape.span(), data.span())
}
