use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3887549, sign: false });
    data.append(FP8x23 { mag: 6278703, sign: false });
    data.append(FP8x23 { mag: 13530644, sign: false });
    data.append(FP8x23 { mag: 7944762, sign: false });
    data.append(FP8x23 { mag: 5571282, sign: false });
    data.append(FP8x23 { mag: 13596525, sign: false });
    data.append(FP8x23 { mag: 6504769, sign: false });
    data.append(FP8x23 { mag: 5900902, sign: false });
    data.append(FP8x23 { mag: 13576015, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
