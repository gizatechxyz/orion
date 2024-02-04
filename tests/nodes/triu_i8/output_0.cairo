use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(10);
    data.append(78);
    data.append(-79);
    data.append(-74);
    data.append(-116);
    data.append(0);
    data.append(77);
    data.append(85);
    data.append(96);
    data.append(-110);
    data.append(0);
    data.append(0);
    data.append(50);
    data.append(-38);
    data.append(117);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(-105);
    data.append(119);
    TensorTrait::new(shape.span(), data.span())
}
