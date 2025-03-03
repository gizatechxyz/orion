use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8324193, sign: false });
    data.append(FP8x23 { mag: 41276796, sign: false });
    data.append(FP8x23 { mag: 11916018, sign: false });
    data.append(FP8x23 { mag: 8708916, sign: true });
    data.append(FP8x23 { mag: 20973208, sign: false });
    data.append(FP8x23 { mag: 439772, sign: false });
    data.append(FP8x23 { mag: 1109136, sign: false });
    data.append(FP8x23 { mag: 3642192, sign: true });
    data.append(FP8x23 { mag: 4556924, sign: true });
    data.append(FP8x23 { mag: 10662183, sign: false });
    data.append(FP8x23 { mag: 2233377, sign: false });
    data.append(FP8x23 { mag: 415587, sign: false });
    data.append(FP8x23 { mag: 1043878, sign: false });
    data.append(FP8x23 { mag: 761953, sign: true });
    data.append(FP8x23 { mag: 3655090, sign: false });
    data.append(FP8x23 { mag: 5875499, sign: true });
    data.append(FP8x23 { mag: 12392297, sign: true });
    data.append(FP8x23 { mag: 15821156, sign: true });
    data.append(FP8x23 { mag: 13421392, sign: true });
    data.append(FP8x23 { mag: 6879230, sign: true });
    data.append(FP8x23 { mag: 10755278, sign: true });
    data.append(FP8x23 { mag: 31152384, sign: false });
    data.append(FP8x23 { mag: 30310852, sign: false });
    data.append(FP8x23 { mag: 20472572, sign: false });
    data.append(FP8x23 { mag: 2600771, sign: false });
    data.append(FP8x23 { mag: 6900111, sign: false });
    data.append(FP8x23 { mag: 9326585, sign: true });
    data.append(FP8x23 { mag: 2789747, sign: false });
    data.append(FP8x23 { mag: 2066718, sign: false });
    data.append(FP8x23 { mag: 1581980, sign: false });
    data.append(FP8x23 { mag: 1585287, sign: false });
    data.append(FP8x23 { mag: 81403, sign: true });
    data.append(FP8x23 { mag: 1234419, sign: false });
    data.append(FP8x23 { mag: 4014873, sign: false });
    data.append(FP8x23 { mag: 167194, sign: true });
    data.append(FP8x23 { mag: 16841552, sign: true });
    data.append(FP8x23 { mag: 13551288, sign: true });
    data.append(FP8x23 { mag: 6322981, sign: true });
    data.append(FP8x23 { mag: 9937094, sign: true });
    data.append(FP8x23 { mag: 7736661, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
