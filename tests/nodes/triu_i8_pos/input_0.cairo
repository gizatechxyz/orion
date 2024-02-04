use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(79);
    data.append(-37);
    data.append(-89);
    data.append(-85);
    data.append(3);
    data.append(68);
    data.append(42);
    data.append(61);
    data.append(-62);
    data.append(-37);
    data.append(-88);
    data.append(69);
    data.append(-123);
    data.append(43);
    data.append(-92);
    data.append(87);
    data.append(-23);
    data.append(-25);
    data.append(101);
    data.append(4);
    TensorTrait::new(shape.span(), data.span())
}
