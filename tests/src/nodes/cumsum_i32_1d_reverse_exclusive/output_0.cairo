use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 14, sign: false });
    data.append(i32 { mag: 12, sign: false });
    data.append(i32 { mag: 9, sign: false });
    data.append(i32 { mag: 5, sign: false });
    data.append(i32 { mag: 0, sign: false });
    TensorTrait::new(shape.span(), data.span())
}