use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6291456, sign: false });
    data.append(FP16x16 { mag: 6422528, sign: true });
    data.append(FP16x16 { mag: 7405568, sign: true });
    data.append(FP16x16 { mag: 7667712, sign: true });
    data.append(FP16x16 { mag: 2031616, sign: true });
    data.append(FP16x16 { mag: 7536640, sign: false });
    data.append(FP16x16 { mag: 2686976, sign: true });
    data.append(FP16x16 { mag: 1507328, sign: true });
    data.append(FP16x16 { mag: 4980736, sign: false });
    data.append(FP16x16 { mag: 1179648, sign: false });
    data.append(FP16x16 { mag: 6815744, sign: false });
    data.append(FP16x16 { mag: 7143424, sign: true });
    data.append(FP16x16 { mag: 2424832, sign: true });
    data.append(FP16x16 { mag: 5636096, sign: false });
    data.append(FP16x16 { mag: 131072, sign: true });
    data.append(FP16x16 { mag: 1507328, sign: false });
    data.append(FP16x16 { mag: 5898240, sign: false });
    data.append(FP16x16 { mag: 6750208, sign: false });
    data.append(FP16x16 { mag: 2621440, sign: false });
    data.append(FP16x16 { mag: 2949120, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
