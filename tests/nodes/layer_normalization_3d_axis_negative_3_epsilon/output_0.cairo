use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 50215, sign: false });
    data.append(FP16x16 { mag: 236376, sign: false });
    data.append(FP16x16 { mag: 71242, sign: true });
    data.append(FP16x16 { mag: 30342, sign: true });
    data.append(FP16x16 { mag: 88310, sign: true });
    data.append(FP16x16 { mag: 20655, sign: true });
    data.append(FP16x16 { mag: 6794, sign: true });
    data.append(FP16x16 { mag: 67191, sign: false });
    data.append(FP16x16 { mag: 235574, sign: false });
    data.append(FP16x16 { mag: 38362, sign: true });
    data.append(FP16x16 { mag: 40541, sign: true });
    data.append(FP16x16 { mag: 10558, sign: true });
    data.append(FP16x16 { mag: 138052, sign: false });
    data.append(FP16x16 { mag: 109426, sign: true });
    data.append(FP16x16 { mag: 56553, sign: true });
    data.append(FP16x16 { mag: 98633, sign: true });
    data.append(FP16x16 { mag: 67749, sign: false });
    data.append(FP16x16 { mag: 32554, sign: true });
    data.append(FP16x16 { mag: 19947, sign: true });
    data.append(FP16x16 { mag: 79300, sign: true });
    data.append(FP16x16 { mag: 63887, sign: false });
    data.append(FP16x16 { mag: 34057, sign: false });
    data.append(FP16x16 { mag: 104451, sign: true });
    data.append(FP16x16 { mag: 54136, sign: true });
    data.append(FP16x16 { mag: 44689, sign: true });
    data.append(FP16x16 { mag: 86047, sign: true });
    data.append(FP16x16 { mag: 119451, sign: true });
    data.append(FP16x16 { mag: 108647, sign: false });
    data.append(FP16x16 { mag: 106358, sign: false });
    data.append(FP16x16 { mag: 47435, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
