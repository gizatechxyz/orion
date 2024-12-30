use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(12220);
    data.append(23899);
    data.append(10958);
    data.append(-5890);
    data.append(319);
    data.append(18006);
    data.append(4962);
    data.append(585);
    data.append(-11930);
    TensorTrait::new(shape.span(), data.span())
}
