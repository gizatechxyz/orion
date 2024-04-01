use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 58066, sign: false });
    data.append(FP16x16 { mag: 62993, sign: false });
    data.append(FP16x16 { mag: 60777, sign: false });
    data.append(FP16x16 { mag: 62324, sign: false });
    data.append(FP16x16 { mag: 63180, sign: true });
    data.append(FP16x16 { mag: 32469, sign: true });
    data.append(FP16x16 { mag: 32381, sign: false });
    data.append(FP16x16 { mag: 19706, sign: true });
    data.append(FP16x16 { mag: 61231, sign: false });
    data.append(FP16x16 { mag: 60506, sign: false });
    data.append(FP16x16 { mag: 63857, sign: false });
    data.append(FP16x16 { mag: 58566, sign: false });
    data.append(FP16x16 { mag: 10803, sign: true });
    data.append(FP16x16 { mag: 54490, sign: true });
    data.append(FP16x16 { mag: 31412, sign: false });
    data.append(FP16x16 { mag: 49093, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
