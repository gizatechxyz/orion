use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(6378);
    data.append(-7461);
    data.append(4251);
    data.append(-6250);
    data.append(6620);
    data.append(-3808);
    data.append(-2734);
    data.append(-10027);
    data.append(5021);
    TensorTrait::new(shape.span(), data.span())
}
