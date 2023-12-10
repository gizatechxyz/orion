use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 74, sign: true });
    data.append(i32 { mag: 122, sign: false });
    data.append(i32 { mag: 108, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
