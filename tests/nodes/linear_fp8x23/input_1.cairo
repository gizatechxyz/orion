use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorDiv};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 53179762, sign: false });
    data.append(FP8x23 { mag: 34716255, sign: false });
    data.append(FP8x23 { mag: 6923297, sign: false });
    data.append(FP8x23 { mag: 2365083, sign: true });
    data.append(FP8x23 { mag: 35449593, sign: true });
    data.append(FP8x23 { mag: 46253431, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
