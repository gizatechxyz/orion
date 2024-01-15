use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 251658240, sign: true });
    data.append(FP8x23 { mag: 864026624, sign: false });
    data.append(FP8x23 { mag: 1006632960, sign: true });
    data.append(FP8x23 { mag: 8388608, sign: false });
    data.append(FP8x23 { mag: 889192448, sign: true });
    data.append(FP8x23 { mag: 243269632, sign: false });
    data.append(FP8x23 { mag: 402653184, sign: false });
    data.append(FP8x23 { mag: 822083584, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
