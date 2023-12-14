use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 5136, sign: true });
    data.append(i32 { mag: 15456, sign: false });
    data.append(i32 { mag: 3518, sign: true });
    data.append(i32 { mag: 10483, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
