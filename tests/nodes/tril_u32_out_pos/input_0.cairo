use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(111);
    data.append(50);
    data.append(147);
    data.append(251);
    data.append(54);
    data.append(8);
    data.append(96);
    data.append(236);
    data.append(214);
    data.append(32);
    data.append(100);
    data.append(220);
    data.append(220);
    data.append(137);
    data.append(66);
    data.append(197);
    data.append(45);
    data.append(126);
    data.append(230);
    data.append(72);
    TensorTrait::new(shape.span(), data.span())
}
