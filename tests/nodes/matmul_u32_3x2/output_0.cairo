use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(43290);
    data.append(39026);
    data.append(49831);
    data.append(40410);
    data.append(42493);
    data.append(49988);
    data.append(28260);
    data.append(22630);
    data.append(30900);
    TensorTrait::new(shape.span(), data.span())
}
