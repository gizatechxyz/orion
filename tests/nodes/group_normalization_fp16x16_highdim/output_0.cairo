use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5632, sign: true });
    data.append(FP16x16 { mag: 36432, sign: false });
    data.append(FP16x16 { mag: 84366, sign: false });
    data.append(FP16x16 { mag: 36600, sign: true });
    data.append(FP16x16 { mag: 171837, sign: false });
    data.append(FP16x16 { mag: 3354, sign: false });
    data.append(FP16x16 { mag: 1202, sign: true });
    data.append(FP16x16 { mag: 45005, sign: false });
    data.append(FP16x16 { mag: 69848, sign: false });
    data.append(FP16x16 { mag: 11411, sign: true });
    data.append(FP16x16 { mag: 28192, sign: true });
    data.append(FP16x16 { mag: 25529, sign: false });
    data.append(FP16x16 { mag: 221137, sign: false });
    data.append(FP16x16 { mag: 92705, sign: false });
    data.append(FP16x16 { mag: 152963, sign: false });
    data.append(FP16x16 { mag: 31712, sign: false });
    data.append(FP16x16 { mag: 72251, sign: false });
    data.append(FP16x16 { mag: 94029, sign: false });
    data.append(FP16x16 { mag: 111699, sign: false });
    data.append(FP16x16 { mag: 104347, sign: false });
    data.append(FP16x16 { mag: 61267, sign: true });
    data.append(FP16x16 { mag: 106458, sign: true });
    data.append(FP16x16 { mag: 8977, sign: false });
    data.append(FP16x16 { mag: 116061, sign: false });
    data.append(FP16x16 { mag: 104129, sign: false });
    data.append(FP16x16 { mag: 19424, sign: false });
    data.append(FP16x16 { mag: 31809, sign: true });
    data.append(FP16x16 { mag: 18010, sign: false });
    data.append(FP16x16 { mag: 26784, sign: false });
    data.append(FP16x16 { mag: 140280, sign: false });
    data.append(FP16x16 { mag: 12780, sign: true });
    data.append(FP16x16 { mag: 19784, sign: false });
    data.append(FP16x16 { mag: 182850, sign: false });
    data.append(FP16x16 { mag: 207568, sign: false });
    data.append(FP16x16 { mag: 42039, sign: false });
    data.append(FP16x16 { mag: 71102, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
