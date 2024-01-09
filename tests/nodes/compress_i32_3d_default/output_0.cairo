use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 9, sign: false });
    data.append(i32 { mag: 10, sign: false });
    data.append(i32 { mag: 11, sign: false });
    data.append(i32 { mag: 12, sign: false });
    data.append(i32 { mag: 13, sign: false });
    data.append(i32 { mag: 14, sign: false });
    data.append(i32 { mag: 15, sign: false });
    data.append(i32 { mag: 16, sign: false });
    data.append(i32 { mag: 17, sign: false });
    data.append(i32 { mag: 18, sign: false });
    data.append(i32 { mag: 19, sign: false });
    data.append(i32 { mag: 20, sign: false });
    data.append(i32 { mag: 21, sign: false });
    data.append(i32 { mag: 22, sign: false });
    data.append(i32 { mag: 23, sign: false });
    data.append(i32 { mag: 24, sign: false });
    data.append(i32 { mag: 25, sign: false });
    data.append(i32 { mag: 26, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
