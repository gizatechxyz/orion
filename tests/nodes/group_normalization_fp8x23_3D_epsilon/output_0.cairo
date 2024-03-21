use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 15706582, sign: true });
    data.append(FP8x23 { mag: 7010898, sign: false });
    data.append(FP8x23 { mag: 6488826, sign: false });
    data.append(FP8x23 { mag: 5917515, sign: false });
    data.append(FP8x23 { mag: 22161104, sign: true });
    data.append(FP8x23 { mag: 11682587, sign: true });
    data.append(FP8x23 { mag: 15795042, sign: false });
    data.append(FP8x23 { mag: 17302248, sign: false });
    data.append(FP8x23 { mag: 5442409, sign: true });
    data.append(FP8x23 { mag: 7226953, sign: false });
    data.append(FP8x23 { mag: 11613764, sign: false });
    data.append(FP8x23 { mag: 4990380, sign: false });
    data.append(FP8x23 { mag: 14249561, sign: true });
    data.append(FP8x23 { mag: 2686082, sign: true });
    data.append(FP8x23 { mag: 17884884, sign: false });
    data.append(FP8x23 { mag: 16846826, sign: false });
    data.append(FP8x23 { mag: 10381656, sign: false });
    data.append(FP8x23 { mag: 266992, sign: false });
    data.append(FP8x23 { mag: 10585391, sign: false });
    data.append(FP8x23 { mag: 9569225, sign: false });
    data.append(FP8x23 { mag: 16337716, sign: true });
    data.append(FP8x23 { mag: 12115941, sign: true });
    data.append(FP8x23 { mag: 17890414, sign: false });
    data.append(FP8x23 { mag: 15727905, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
