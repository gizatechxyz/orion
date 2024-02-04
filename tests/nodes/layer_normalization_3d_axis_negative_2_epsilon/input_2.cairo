use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 18752, sign: false });
    data.append(FP16x16 { mag: 19935, sign: false });
    data.append(FP16x16 { mag: 20182, sign: true });
    data.append(FP16x16 { mag: 65500, sign: false });
    data.append(FP16x16 { mag: 3405, sign: true });
    data.append(FP16x16 { mag: 43973, sign: true });
    data.append(FP16x16 { mag: 45323, sign: true });
    data.append(FP16x16 { mag: 24495, sign: false });
    data.append(FP16x16 { mag: 27923, sign: true });
    data.append(FP16x16 { mag: 149149, sign: false });
    data.append(FP16x16 { mag: 3341, sign: true });
    data.append(FP16x16 { mag: 17501, sign: false });
    data.append(FP16x16 { mag: 38949, sign: true });
    data.append(FP16x16 { mag: 85674, sign: false });
    data.append(FP16x16 { mag: 3401, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
