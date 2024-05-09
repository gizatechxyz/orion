use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 50189060, sign: false });
    data.append(FP8x23 { mag: 19900964, sign: false });
    data.append(FP8x23 { mag: 54019428, sign: false });
    data.append(FP8x23 { mag: 19767152, sign: false });
    data.append(FP8x23 { mag: 18003788, sign: false });
    data.append(FP8x23 { mag: 9743942, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
