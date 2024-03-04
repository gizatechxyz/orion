use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6291456, sign: false });
    data.append(FP16x16 { mag: 720896, sign: true });
    data.append(FP16x16 { mag: 4521984, sign: false });
    data.append(FP16x16 { mag: 5308416, sign: false });
    data.append(FP16x16 { mag: 655360, sign: true });
    data.append(FP16x16 { mag: 1507328, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 6356992, sign: false });
    data.append(FP16x16 { mag: 393216, sign: true });
    data.append(FP16x16 { mag: 3342336, sign: false });
    data.append(FP16x16 { mag: 524288, sign: false });
    data.append(FP16x16 { mag: 2490368, sign: true });
    data.append(FP16x16 { mag: 5308416, sign: false });
    data.append(FP16x16 { mag: 6881280, sign: false });
    data.append(FP16x16 { mag: 3538944, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 6356992, sign: true });
    data.append(FP16x16 { mag: 6684672, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
