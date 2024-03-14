use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 964915, sign: true });
    data.append(FP16x16 { mag: 60080, sign: true });
    data.append(FP16x16 { mag: 282570, sign: false });
    data.append(FP16x16 { mag: 1276847, sign: false });
    data.append(FP16x16 { mag: 680331, sign: true });
    data.append(FP16x16 { mag: 838194, sign: false });
    data.append(FP16x16 { mag: 933879, sign: true });
    data.append(FP16x16 { mag: 667340, sign: false });
    data.append(FP16x16 { mag: 355340, sign: true });
    data.append(FP16x16 { mag: 59919, sign: false });
    data.append(FP16x16 { mag: 942330, sign: false });
    data.append(FP16x16 { mag: 27589, sign: true });
    data.append(FP16x16 { mag: 503258, sign: false });
    data.append(FP16x16 { mag: 1019961, sign: false });
    data.append(FP16x16 { mag: 1074737, sign: false });
    data.append(FP16x16 { mag: 424644, sign: true });
    data.append(FP16x16 { mag: 599690, sign: false });
    data.append(FP16x16 { mag: 1182030, sign: true });
    data.append(FP16x16 { mag: 818383, sign: true });
    data.append(FP16x16 { mag: 1230651, sign: false });
    data.append(FP16x16 { mag: 660681, sign: true });
    data.append(FP16x16 { mag: 774752, sign: true });
    data.append(FP16x16 { mag: 834572, sign: true });
    data.append(FP16x16 { mag: 1029735, sign: true });
    data.append(FP16x16 { mag: 605008, sign: false });
    data.append(FP16x16 { mag: 1295696, sign: false });
    data.append(FP16x16 { mag: 52967, sign: false });
    data.append(FP16x16 { mag: 13759, sign: true });
    data.append(FP16x16 { mag: 937699, sign: false });
    data.append(FP16x16 { mag: 1006488, sign: true });
    data.append(FP16x16 { mag: 889419, sign: true });
    data.append(FP16x16 { mag: 472374, sign: true });
    data.append(FP16x16 { mag: 387220, sign: false });
    data.append(FP16x16 { mag: 562768, sign: true });
    data.append(FP16x16 { mag: 880496, sign: true });
    data.append(FP16x16 { mag: 993454, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
