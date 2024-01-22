use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(-74);
    data.append(7);
    data.append(11);
    data.append(123);
    data.append(-67);
    data.append(30);
    data.append(3);
    data.append(90);
    data.append(-85);
    data.append(111);
    data.append(45);
    data.append(76);
    data.append(60);
    data.append(121);
    data.append(89);
    data.append(114);
    data.append(11);
    data.append(-92);
    data.append(-79);
    data.append(104);
    TensorTrait::new(shape.span(), data.span())
}
