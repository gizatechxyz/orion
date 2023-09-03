use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 26, sign: true });
    data.append(i32 { mag: 22, sign: false });
    data.append(i32 { mag: 28, sign: true });
    data.append(i32 { mag: 82, sign: false });
    data.append(i32 { mag: 45, sign: false });
    data.append(i32 { mag: 62, sign: false });
    data.append(i32 { mag: 27, sign: true });
    data.append(i32 { mag: 54, sign: false });
    TensorTrait::new(shape.span(), data.span())
}