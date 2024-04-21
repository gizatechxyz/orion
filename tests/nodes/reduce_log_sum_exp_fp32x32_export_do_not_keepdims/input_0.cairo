use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP32x32Tensor;
use orion::numbers::{FixedTrait, FP32x32};

fn input_0() -> Tensor<FP32x32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP32x32 { mag: 4294967296, sign: false });
    data.append(FP32x32 { mag: 8589934592, sign: false });
    data.append(FP32x32 { mag: 12884901888, sign: false });
    data.append(FP32x32 { mag: 17179869184, sign: false });
    data.append(FP32x32 { mag: 21474836480, sign: false });
    data.append(FP32x32 { mag: 25769803776, sign: false });
    data.append(FP32x32 { mag: 30064771072, sign: false });
    data.append(FP32x32 { mag: 34359738368, sign: false });
    data.append(FP32x32 { mag: 38654705664, sign: false });
    data.append(FP32x32 { mag: 42949672960, sign: false });
    data.append(FP32x32 { mag: 47244640256, sign: false });
    data.append(FP32x32 { mag: 51539607552, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
