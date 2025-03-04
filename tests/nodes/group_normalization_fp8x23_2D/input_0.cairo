use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 12194073, sign: true });
    data.append(FP8x23 { mag: 13381644, sign: true });
    data.append(FP8x23 { mag: 10647423, sign: true });
    data.append(FP8x23 { mag: 6142784, sign: true });
    data.append(FP8x23 { mag: 9774141, sign: false });
    data.append(FP8x23 { mag: 5325817, sign: true });
    data.append(FP8x23 { mag: 9436893, sign: false });
    data.append(FP8x23 { mag: 881873, sign: true });
    data.append(FP8x23 { mag: 11936498, sign: true });
    data.append(FP8x23 { mag: 9427479, sign: true });
    data.append(FP8x23 { mag: 3320860, sign: true });
    data.append(FP8x23 { mag: 13210179, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
