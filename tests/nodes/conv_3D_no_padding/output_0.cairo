use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 54853632, sign: false });
    data.append(FP16x16 { mag: 56623104, sign: false });
    data.append(FP16x16 { mag: 58392576, sign: false });
    data.append(FP16x16 { mag: 63700992, sign: false });
    data.append(FP16x16 { mag: 65470464, sign: false });
    data.append(FP16x16 { mag: 67239936, sign: false });
    data.append(FP16x16 { mag: 72548352, sign: false });
    data.append(FP16x16 { mag: 74317824, sign: false });
    data.append(FP16x16 { mag: 76087296, sign: false });
    data.append(FP16x16 { mag: 99090432, sign: false });
    data.append(FP16x16 { mag: 100859904, sign: false });
    data.append(FP16x16 { mag: 102629376, sign: false });
    data.append(FP16x16 { mag: 107937792, sign: false });
    data.append(FP16x16 { mag: 109707264, sign: false });
    data.append(FP16x16 { mag: 111476736, sign: false });
    data.append(FP16x16 { mag: 116785152, sign: false });
    data.append(FP16x16 { mag: 118554624, sign: false });
    data.append(FP16x16 { mag: 120324096, sign: false });
    data.append(FP16x16 { mag: 143327232, sign: false });
    data.append(FP16x16 { mag: 145096704, sign: false });
    data.append(FP16x16 { mag: 146866176, sign: false });
    data.append(FP16x16 { mag: 152174592, sign: false });
    data.append(FP16x16 { mag: 153944064, sign: false });
    data.append(FP16x16 { mag: 155713536, sign: false });
    data.append(FP16x16 { mag: 161021952, sign: false });
    data.append(FP16x16 { mag: 162791424, sign: false });
    data.append(FP16x16 { mag: 164560896, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
