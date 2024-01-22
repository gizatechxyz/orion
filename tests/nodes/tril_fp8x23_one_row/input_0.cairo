use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 671088640, sign: false });
    data.append(FP8x23 { mag: 905969664, sign: true });
    data.append(FP8x23 { mag: 41943040, sign: false });
    data.append(FP8x23 { mag: 889192448, sign: false });
    data.append(FP8x23 { mag: 25165824, sign: true });
    data.append(FP8x23 { mag: 679477248, sign: true });
    data.append(FP8x23 { mag: 461373440, sign: true });
    data.append(FP8x23 { mag: 142606336, sign: false });
    data.append(FP8x23 { mag: 796917760, sign: true });
    data.append(FP8x23 { mag: 511705088, sign: false });
    data.append(FP8x23 { mag: 100663296, sign: false });
    data.append(FP8x23 { mag: 947912704, sign: false });
    data.append(FP8x23 { mag: 461373440, sign: false });
    data.append(FP8x23 { mag: 92274688, sign: false });
    data.append(FP8x23 { mag: 956301312, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
