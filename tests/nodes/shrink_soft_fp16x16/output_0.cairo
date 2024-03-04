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
    data.append(FP16x16 { mag: 130914, sign: false });
    data.append(FP16x16 { mag: 44922, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 22733, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 76015, sign: true });
    data.append(FP16x16 { mag: 70901, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 91946, sign: false });
    data.append(FP16x16 { mag: 114938, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 25171, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 32819, sign: false });
    data.append(FP16x16 { mag: 60011, sign: false });
    data.append(FP16x16 { mag: 43323, sign: true });
    data.append(FP16x16 { mag: 54975, sign: true });
    data.append(FP16x16 { mag: 43594, sign: true });
    data.append(FP16x16 { mag: 76123, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 5713, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
