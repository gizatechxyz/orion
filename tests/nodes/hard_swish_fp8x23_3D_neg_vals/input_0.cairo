use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 46028652, sign: true });
    data.append(FP8x23 { mag: 43733988, sign: true });
    data.append(FP8x23 { mag: 50069680, sign: true });
    data.append(FP8x23 { mag: 49305588, sign: true });
    data.append(FP8x23 { mag: 43817324, sign: true });
    data.append(FP8x23 { mag: 47993856, sign: true });
    data.append(FP8x23 { mag: 42874448, sign: true });
    data.append(FP8x23 { mag: 35855144, sign: true });
    data.append(FP8x23 { mag: 32744554, sign: true });
    data.append(FP8x23 { mag: 28256186, sign: true });
    data.append(FP8x23 { mag: 32322936, sign: true });
    data.append(FP8x23 { mag: 28468444, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
