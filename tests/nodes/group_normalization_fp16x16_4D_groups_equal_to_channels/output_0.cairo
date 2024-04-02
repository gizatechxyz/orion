use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 92156, sign: true });
    data.append(FP16x16 { mag: 136576, sign: true });
    data.append(FP16x16 { mag: 132977, sign: true });
    data.append(FP16x16 { mag: 26775, sign: false });
    data.append(FP16x16 { mag: 209242, sign: true });
    data.append(FP16x16 { mag: 66196, sign: false });
    data.append(FP16x16 { mag: 99992, sign: false });
    data.append(FP16x16 { mag: 66193, sign: false });
    data.append(FP16x16 { mag: 2266, sign: false });
    data.append(FP16x16 { mag: 159120, sign: true });
    data.append(FP16x16 { mag: 94615, sign: true });
    data.append(FP16x16 { mag: 83465, sign: true });
    data.append(FP16x16 { mag: 148047, sign: true });
    data.append(FP16x16 { mag: 192406, sign: false });
    data.append(FP16x16 { mag: 37789, sign: false });
    data.append(FP16x16 { mag: 59007, sign: true });
    data.append(FP16x16 { mag: 144697, sign: true });
    data.append(FP16x16 { mag: 8692, sign: false });
    data.append(FP16x16 { mag: 55796, sign: true });
    data.append(FP16x16 { mag: 143134, sign: true });
    data.append(FP16x16 { mag: 153417, sign: false });
    data.append(FP16x16 { mag: 49782, sign: false });
    data.append(FP16x16 { mag: 139126, sign: true });
    data.append(FP16x16 { mag: 40932, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
