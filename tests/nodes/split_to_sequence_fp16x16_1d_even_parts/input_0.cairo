use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(18);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 4259840, sign: true });
    data.append(FP16x16 { mag: 2555904, sign: true });
    data.append(FP16x16 { mag: 5767168, sign: true });
    data.append(FP16x16 { mag: 3538944, sign: false });
    data.append(FP16x16 { mag: 131072, sign: true });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 4063232, sign: false });
    data.append(FP16x16 { mag: 1441792, sign: true });
    data.append(FP16x16 { mag: 4849664, sign: false });
    data.append(FP16x16 { mag: 1441792, sign: false });
    data.append(FP16x16 { mag: 4063232, sign: false });
    data.append(FP16x16 { mag: 2424832, sign: false });
    data.append(FP16x16 { mag: 6225920, sign: true });
    data.append(FP16x16 { mag: 3276800, sign: false });
    data.append(FP16x16 { mag: 1114112, sign: true });
    data.append(FP16x16 { mag: 786432, sign: false });
    data.append(FP16x16 { mag: 7864320, sign: true });
    data.append(FP16x16 { mag: 5570560, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
