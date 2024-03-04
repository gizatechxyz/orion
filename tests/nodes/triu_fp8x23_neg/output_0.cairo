use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 25165824, sign: true });
    data.append(FP8x23 { mag: 713031680, sign: true });
    data.append(FP8x23 { mag: 1040187392, sign: true });
    data.append(FP8x23 { mag: 251658240, sign: false });
    data.append(FP8x23 { mag: 67108864, sign: true });
    data.append(FP8x23 { mag: 117440512, sign: true });
    data.append(FP8x23 { mag: 461373440, sign: false });
    data.append(FP8x23 { mag: 360710144, sign: false });
    data.append(FP8x23 { mag: 486539264, sign: false });
    data.append(FP8x23 { mag: 1006632960, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 478150656, sign: true });
    data.append(FP8x23 { mag: 981467136, sign: true });
    data.append(FP8x23 { mag: 234881024, sign: true });
    data.append(FP8x23 { mag: 377487360, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 645922816, sign: false });
    data.append(FP8x23 { mag: 444596224, sign: true });
    data.append(FP8x23 { mag: 41943040, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
