use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(15);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 60630, sign: false });
    data.append(FP16x16 { mag: 63657, sign: false });
    data.append(FP16x16 { mag: 65797, sign: false });
    data.append(FP16x16 { mag: 136693, sign: false });
    data.append(FP16x16 { mag: 132432, sign: false });
    data.append(FP16x16 { mag: 99730, sign: false });
    data.append(FP16x16 { mag: 104964, sign: false });
    data.append(FP16x16 { mag: 71101, sign: false });
    data.append(FP16x16 { mag: 58060, sign: false });
    data.append(FP16x16 { mag: 15360, sign: false });
    data.append(FP16x16 { mag: 15374, sign: false });
    data.append(FP16x16 { mag: 18850, sign: false });
    data.append(FP16x16 { mag: 70770, sign: false });
    data.append(FP16x16 { mag: 94075, sign: false });
    data.append(FP16x16 { mag: 95924, sign: false });
    data.append(FP16x16 { mag: 29523, sign: false });
    data.append(FP16x16 { mag: 29661, sign: false });
    data.append(FP16x16 { mag: 51817, sign: false });
    data.append(FP16x16 { mag: 54472, sign: false });
    data.append(FP16x16 { mag: 80319, sign: false });
    data.append(FP16x16 { mag: 79106, sign: false });
    data.append(FP16x16 { mag: 57514, sign: false });
    data.append(FP16x16 { mag: 80945, sign: false });
    data.append(FP16x16 { mag: 69883, sign: false });
    data.append(FP16x16 { mag: 43738, sign: false });
    data.append(FP16x16 { mag: 44655, sign: false });
    data.append(FP16x16 { mag: 63578, sign: false });
    data.append(FP16x16 { mag: 69666, sign: false });
    data.append(FP16x16 { mag: 77484, sign: false });
    data.append(FP16x16 { mag: 72281, sign: false });
    data.append(FP16x16 { mag: 48489, sign: false });
    data.append(FP16x16 { mag: 111977, sign: false });
    data.append(FP16x16 { mag: 112563, sign: false });
    data.append(FP16x16 { mag: 81044, sign: false });
    data.append(FP16x16 { mag: 80558, sign: false });
    data.append(FP16x16 { mag: 23410, sign: false });
    data.append(FP16x16 { mag: 41683, sign: false });
    data.append(FP16x16 { mag: 52844, sign: false });
    data.append(FP16x16 { mag: 69945, sign: false });
    data.append(FP16x16 { mag: 106426, sign: false });
    data.append(FP16x16 { mag: 99549, sign: false });
    data.append(FP16x16 { mag: 108867, sign: false });
    data.append(FP16x16 { mag: 107752, sign: false });
    data.append(FP16x16 { mag: 79932, sign: false });
    data.append(FP16x16 { mag: 79635, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
