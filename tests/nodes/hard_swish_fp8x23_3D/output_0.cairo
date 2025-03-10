use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 4034671, sign: false });
    data.append(FP8x23 { mag: 3145709, sign: true });
    data.append(FP8x23 { mag: 21437444, sign: false });
    data.append(FP8x23 { mag: 1972955, sign: true });
    data.append(FP8x23 { mag: 5113328, sign: false });
    data.append(FP8x23 { mag: 19912274, sign: false });
    data.append(FP8x23 { mag: 1305081, sign: true });
    data.append(FP8x23 { mag: 849901, sign: false });
    data.append(FP8x23 { mag: 6889909, sign: false });
    data.append(FP8x23 { mag: 4066779, sign: false });
    data.append(FP8x23 { mag: 2049274, sign: true });
    data.append(FP8x23 { mag: 14357457, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
