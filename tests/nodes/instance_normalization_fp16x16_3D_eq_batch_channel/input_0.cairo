use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 77953, sign: true });
    data.append(FP16x16 { mag: 90788, sign: false });
    data.append(FP16x16 { mag: 60665, sign: true });
    data.append(FP16x16 { mag: 161420, sign: false });
    data.append(FP16x16 { mag: 117803, sign: true });
    data.append(FP16x16 { mag: 8294, sign: true });
    data.append(FP16x16 { mag: 62509, sign: false });
    data.append(FP16x16 { mag: 82682, sign: false });
    data.append(FP16x16 { mag: 17836, sign: true });
    data.append(FP16x16 { mag: 44675, sign: false });
    data.append(FP16x16 { mag: 25750, sign: true });
    data.append(FP16x16 { mag: 19300, sign: false });
    data.append(FP16x16 { mag: 6538, sign: false });
    data.append(FP16x16 { mag: 131324, sign: true });
    data.append(FP16x16 { mag: 110452, sign: false });
    data.append(FP16x16 { mag: 114639, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
