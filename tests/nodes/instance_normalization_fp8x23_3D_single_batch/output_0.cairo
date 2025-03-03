use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 26223756, sign: false });
    data.append(FP8x23 { mag: 36704184, sign: false });
    data.append(FP8x23 { mag: 9303670, sign: true });
    data.append(FP8x23 { mag: 6191044, sign: true });
    data.append(FP8x23 { mag: 14603878, sign: false });
    data.append(FP8x23 { mag: 4363463, sign: true });
    data.append(FP8x23 { mag: 334204, sign: false });
    data.append(FP8x23 { mag: 5199698, sign: true });
    data.append(FP8x23 { mag: 5136649, sign: true });
    data.append(FP8x23 { mag: 7880067, sign: true });
    data.append(FP8x23 { mag: 1948209, sign: false });
    data.append(FP8x23 { mag: 13867496, sign: true });
    data.append(FP8x23 { mag: 4912022, sign: true });
    data.append(FP8x23 { mag: 12838364, sign: true });
    data.append(FP8x23 { mag: 5072270, sign: true });
    data.append(FP8x23 { mag: 1950714, sign: false });
    data.append(FP8x23 { mag: 306832, sign: false });
    data.append(FP8x23 { mag: 13751253, sign: true });
    data.append(FP8x23 { mag: 3719595, sign: false });
    data.append(FP8x23 { mag: 8528553, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
