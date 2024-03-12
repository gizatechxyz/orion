use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 40960207, sign: false });
    data.append(FP8x23 { mag: 31287372, sign: true });
    data.append(FP8x23 { mag: 75603722, sign: false });
    data.append(FP8x23 { mag: 139009462, sign: true });
    data.append(FP8x23 { mag: 176439012, sign: true });
    data.append(FP8x23 { mag: 72460509, sign: false });
    data.append(FP8x23 { mag: 54936798, sign: true });
    data.append(FP8x23 { mag: 22294840, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
