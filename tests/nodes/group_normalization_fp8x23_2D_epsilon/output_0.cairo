use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 10113120, sign: true });
    data.append(FP8x23 { mag: 4918010, sign: true });
    data.append(FP8x23 { mag: 2619546, sign: true });
    data.append(FP8x23 { mag: 3955292, sign: false });
    data.append(FP8x23 { mag: 5208441, sign: true });
    data.append(FP8x23 { mag: 3055863, sign: true });
    data.append(FP8x23 { mag: 1447300, sign: true });
    data.append(FP8x23 { mag: 3301988, sign: false });
    data.append(FP8x23 { mag: 9066985, sign: true });
    data.append(FP8x23 { mag: 4520826, sign: true });
    data.append(FP8x23 { mag: 16986818, sign: true });
    data.append(FP8x23 { mag: 11962312, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
