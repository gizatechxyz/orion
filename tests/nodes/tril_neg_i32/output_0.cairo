use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 30, sign: true });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 72, sign: true });
    data.append(i32 { mag: 123, sign: true });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 109, sign: false });
    data.append(i32 { mag: 84, sign: true });
    data.append(i32 { mag: 117, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
