use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(3);
    shape.append(2);
    shape.append(1);
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 5810993, sign: true });
    data.append(FP8x23 { mag: 6045082, sign: true });
    data.append(FP8x23 { mag: 5357330, sign: true });
    data.append(FP8x23 { mag: 5685945, sign: true });
    data.append(FP8x23 { mag: 5360141, sign: true });
    data.append(FP8x23 { mag: 5917737, sign: true });
    data.append(FP8x23 { mag: 24609204, sign: false });
    data.append(FP8x23 { mag: 7347798, sign: false });
    data.append(FP8x23 { mag: 3557345, sign: false });
    data.append(FP8x23 { mag: 1917810, sign: true });
    data.append(FP8x23 { mag: 42682240, sign: false });
    data.append(FP8x23 { mag: 2869398, sign: false });
    data.append(FP8x23 { mag: 5415499, sign: true });
    data.append(FP8x23 { mag: 5985082, sign: true });
    data.append(FP8x23 { mag: 5913520, sign: true });
    data.append(FP8x23 { mag: 5653556, sign: true });
    data.append(FP8x23 { mag: 5907971, sign: true });
    data.append(FP8x23 { mag: 5301599, sign: true });
    data.append(FP8x23 { mag: 14974408, sign: false });
    data.append(FP8x23 { mag: 1892416, sign: true });
    data.append(FP8x23 { mag: 21681716, sign: false });
    data.append(FP8x23 { mag: 40092848, sign: false });
    data.append(FP8x23 { mag: 11928262, sign: false });
    data.append(FP8x23 { mag: 7636644, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
