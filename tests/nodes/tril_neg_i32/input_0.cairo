use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 67, sign: false });
    data.append(i32 { mag: 102, sign: false });
    data.append(i32 { mag: 25, sign: false });
    data.append(i32 { mag: 75, sign: false });
    data.append(i32 { mag: 86, sign: false });
    data.append(i32 { mag: 30, sign: true });
    data.append(i32 { mag: 25, sign: false });
    data.append(i32 { mag: 124, sign: false });
    data.append(i32 { mag: 58, sign: false });
    data.append(i32 { mag: 72, sign: true });
    data.append(i32 { mag: 72, sign: true });
    data.append(i32 { mag: 123, sign: true });
    data.append(i32 { mag: 6, sign: false });
    data.append(i32 { mag: 18, sign: false });
    data.append(i32 { mag: 18, sign: false });
    data.append(i32 { mag: 109, sign: false });
    data.append(i32 { mag: 84, sign: true });
    data.append(i32 { mag: 117, sign: false });
    data.append(i32 { mag: 123, sign: false });
    data.append(i32 { mag: 44, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
