use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 12114806, sign: false });
    data.append(FP8x23 { mag: 11080345, sign: false });
    data.append(FP8x23 { mag: 10258564, sign: false });
    data.append(FP8x23 { mag: 5552570, sign: false });
    data.append(FP8x23 { mag: 4089164, sign: true });
    data.append(FP8x23 { mag: 1081624, sign: true });
    data.append(FP8x23 { mag: 2241284, sign: true });
    data.append(FP8x23 { mag: 3869510, sign: true });
    data.append(FP8x23 { mag: 11739620, sign: false });
    data.append(FP8x23 { mag: 3405714, sign: true });
    data.append(FP8x23 { mag: 5026212, sign: true });
    data.append(FP8x23 { mag: 3592510, sign: false });
    data.append(FP8x23 { mag: 7273391, sign: false });
    data.append(FP8x23 { mag: 11001664, sign: false });
    data.append(FP8x23 { mag: 4634698, sign: false });
    data.append(FP8x23 { mag: 6027630, sign: false });
    data.append(FP8x23 { mag: 4411288, sign: true });
    data.append(FP8x23 { mag: 3305152, sign: true });
    data.append(FP8x23 { mag: 2696765, sign: true });
    data.append(FP8x23 { mag: 4935632, sign: true });
    data.append(FP8x23 { mag: 9807912, sign: false });
    data.append(FP8x23 { mag: 3100167, sign: true });
    data.append(FP8x23 { mag: 10904008, sign: false });
    data.append(FP8x23 { mag: 5455394, sign: true });
    data.append(FP8x23 { mag: 8971461, sign: false });
    data.append(FP8x23 { mag: 9410990, sign: false });
    data.append(FP8x23 { mag: 9901626, sign: false });
    data.append(FP8x23 { mag: 9595503, sign: false });
    data.append(FP8x23 { mag: 4170828, sign: true });
    data.append(FP8x23 { mag: 1416943, sign: true });
    data.append(FP8x23 { mag: 5352173, sign: true });
    data.append(FP8x23 { mag: 835744, sign: true });
    data.append(FP8x23 { mag: 4617051, sign: false });
    data.append(FP8x23 { mag: 7200964, sign: true });
    data.append(FP8x23 { mag: 6982158, sign: false });
    data.append(FP8x23 { mag: 2841996, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
