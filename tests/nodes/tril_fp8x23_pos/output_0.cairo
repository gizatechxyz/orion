use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 377487360, sign: false });
    data.append(FP8x23 { mag: 184549376, sign: true });
    data.append(FP8x23 { mag: 461373440, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 209715200, sign: true });
    data.append(FP8x23 { mag: 989855744, sign: false });
    data.append(FP8x23 { mag: 620756992, sign: true });
    data.append(FP8x23 { mag: 243269632, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 1056964608, sign: false });
    data.append(FP8x23 { mag: 570425344, sign: true });
    data.append(FP8x23 { mag: 33554432, sign: true });
    data.append(FP8x23 { mag: 612368384, sign: false });
    data.append(FP8x23 { mag: 150994944, sign: true });
    data.append(FP8x23 { mag: 872415232, sign: false });
    data.append(FP8x23 { mag: 813694976, sign: false });
    data.append(FP8x23 { mag: 335544320, sign: false });
    data.append(FP8x23 { mag: 796917760, sign: true });
    data.append(FP8x23 { mag: 864026624, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
