use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 19405045, sign: false });
    data.append(FP8x23 { mag: 36182261, sign: false });
    data.append(FP8x23 { mag: 52959477, sign: false });
    data.append(FP8x23 { mag: 69736693, sign: false });
    data.append(FP8x23 { mag: 86513909, sign: false });
    data.append(FP8x23 { mag: 103291125, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
