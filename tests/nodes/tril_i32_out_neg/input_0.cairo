use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 83, sign: true });
    data.append(i32 { mag: 36, sign: true });
    data.append(i32 { mag: 90, sign: true });
    data.append(i32 { mag: 12, sign: true });
    data.append(i32 { mag: 24, sign: false });
    data.append(i32 { mag: 114, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 119, sign: true });
    data.append(i32 { mag: 8, sign: false });
    data.append(i32 { mag: 52, sign: false });
    data.append(i32 { mag: 77, sign: true });
    data.append(i32 { mag: 40, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 93, sign: true });
    data.append(i32 { mag: 71, sign: false });
    data.append(i32 { mag: 123, sign: true });
    data.append(i32 { mag: 107, sign: false });
    data.append(i32 { mag: 63, sign: false });
    data.append(i32 { mag: 100, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
