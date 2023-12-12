use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 38638, sign: false });
    data.append(FP16x16 { mag: 132210, sign: false });
    data.append(FP16x16 { mag: 167838, sign: false });
    data.append(FP16x16 { mag: 48522, sign: false });
    data.append(FP16x16 { mag: 90060, sign: false });
    data.append(FP16x16 { mag: 182305, sign: false });
    data.append(FP16x16 { mag: 195761, sign: false });
    data.append(FP16x16 { mag: 55113, sign: false });
    data.append(FP16x16 { mag: 2489, sign: false });
    data.append(FP16x16 { mag: 12496, sign: false });
    data.append(FP16x16 { mag: 56991, sign: false });
    data.append(FP16x16 { mag: 10365, sign: false });
    data.append(FP16x16 { mag: 80275, sign: false });
    data.append(FP16x16 { mag: 52009, sign: false });
    data.append(FP16x16 { mag: 129459, sign: false });
    data.append(FP16x16 { mag: 36318, sign: false });
    data.append(FP16x16 { mag: 108733, sign: false });
    data.append(FP16x16 { mag: 114068, sign: false });
    data.append(FP16x16 { mag: 89937, sign: false });
    data.append(FP16x16 { mag: 82549, sign: false });
    data.append(FP16x16 { mag: 171965, sign: false });
    data.append(FP16x16 { mag: 120808, sign: false });
    data.append(FP16x16 { mag: 94584, sign: false });
    data.append(FP16x16 { mag: 122438, sign: false });
    data.append(FP16x16 { mag: 147487, sign: false });
    data.append(FP16x16 { mag: 177155, sign: false });
    data.append(FP16x16 { mag: 71361, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
