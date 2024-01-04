use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 43, sign: true });
    data.append(i32 { mag: 24, sign: true });
    data.append(i32 { mag: 63, sign: false });
    data.append(i32 { mag: 117, sign: false });
    data.append(i32 { mag: 101, sign: true });
    data.append(i32 { mag: 90, sign: false });
    data.append(i32 { mag: 50, sign: false });
    data.append(i32 { mag: 66, sign: false });
    data.append(i32 { mag: 120, sign: true });
    data.append(i32 { mag: 15, sign: true });
    data.append(i32 { mag: 24, sign: true });
    data.append(i32 { mag: 123, sign: true });
    data.append(i32 { mag: 16, sign: true });
    data.append(i32 { mag: 96, sign: false });
    data.append(i32 { mag: 33, sign: false });
    data.append(i32 { mag: 63, sign: true });
    data.append(i32 { mag: 117, sign: true });
    data.append(i32 { mag: 52, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
