use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 3211264, sign: false });
    data.append(FP16x16 { mag: 1900544, sign: false });
    data.append(FP16x16 { mag: 3866624, sign: false });
    data.append(FP16x16 { mag: 3145728, sign: false });
    data.append(FP16x16 { mag: 6422528, sign: true });
    data.append(FP16x16 { mag: 3342336, sign: false });
    data.append(FP16x16 { mag: 6553600, sign: true });
    data.append(FP16x16 { mag: 6422528, sign: true });
    data.append(FP16x16 { mag: 4063232, sign: false });
    data.append(FP16x16 { mag: 4521984, sign: false });
    data.append(FP16x16 { mag: 4784128, sign: false });
    data.append(FP16x16 { mag: 5046272, sign: true });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 4521984, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
