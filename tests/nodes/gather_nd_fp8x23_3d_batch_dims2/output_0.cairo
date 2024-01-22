use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16777216, sign: false });
    data.append(FP8x23 { mag: 75497472, sign: false });
    data.append(FP8x23 { mag: 100663296, sign: false });
    data.append(FP8x23 { mag: 159383552, sign: false });
    data.append(FP8x23 { mag: 226492416, sign: false });
    data.append(FP8x23 { mag: 276824064, sign: false });
    data.append(FP8x23 { mag: 310378496, sign: false });
    data.append(FP8x23 { mag: 352321536, sign: false });
    data.append(FP8x23 { mag: 402653184, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
