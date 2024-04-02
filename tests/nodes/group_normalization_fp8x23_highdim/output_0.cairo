use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);
    shape.append(1);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2883275, sign: true });
    data.append(FP8x23 { mag: 4880812, sign: false });
    data.append(FP8x23 { mag: 695982, sign: true });
    data.append(FP8x23 { mag: 3747968, sign: true });
    data.append(FP8x23 { mag: 20298042, sign: true });
    data.append(FP8x23 { mag: 14261491, sign: true });
    data.append(FP8x23 { mag: 7433454, sign: true });
    data.append(FP8x23 { mag: 2949550, sign: false });
    data.append(FP8x23 { mag: 2291173, sign: true });
    data.append(FP8x23 { mag: 23237090, sign: true });
    data.append(FP8x23 { mag: 2232044, sign: false });
    data.append(FP8x23 { mag: 10986772, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
