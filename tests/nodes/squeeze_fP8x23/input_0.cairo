use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 545259520, sign: false });
    data.append(FP8x23 { mag: 897581056, sign: false });
    data.append(FP8x23 { mag: 1367343104, sign: false });
    data.append(FP8x23 { mag: 226492416, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
