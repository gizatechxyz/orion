use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(46);
    data.append(42);
    data.append(72);
    data.append(75);
    data.append(4);
    data.append(14);
    TensorTrait::new(shape.span(), data.span())
}
