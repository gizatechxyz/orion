use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 38624, sign: false });
    data.append(FP16x16 { mag: 46848, sign: true });
    data.append(FP16x16 { mag: 183424, sign: true });
    data.append(FP16x16 { mag: 60512, sign: true });
    data.append(FP16x16 { mag: 134400, sign: false });
    data.append(FP16x16 { mag: 10776, sign: false });
    data.append(FP16x16 { mag: 111104, sign: true });
    data.append(FP16x16 { mag: 155776, sign: false });
    data.append(FP16x16 { mag: 50592, sign: true });
    data.append(FP16x16 { mag: 175872, sign: false });
    data.append(FP16x16 { mag: 23344, sign: true });
    data.append(FP16x16 { mag: 149376, sign: true });
    data.append(FP16x16 { mag: 187648, sign: false });
    data.append(FP16x16 { mag: 180480, sign: true });
    data.append(FP16x16 { mag: 143616, sign: true });
    data.append(FP16x16 { mag: 108608, sign: true });
    data.append(FP16x16 { mag: 129088, sign: false });
    data.append(FP16x16 { mag: 180096, sign: false });
    data.append(FP16x16 { mag: 184192, sign: false });
    data.append(FP16x16 { mag: 134656, sign: true });
    data.append(FP16x16 { mag: 147328, sign: false });
    data.append(FP16x16 { mag: 117120, sign: true });
    data.append(FP16x16 { mag: 50528, sign: false });
    data.append(FP16x16 { mag: 134528, sign: false });
    data.append(FP16x16 { mag: 115712, sign: false });
    data.append(FP16x16 { mag: 92672, sign: false });
    data.append(FP16x16 { mag: 78592, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
