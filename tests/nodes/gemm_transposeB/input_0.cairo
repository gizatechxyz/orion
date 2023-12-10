use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 44827, sign: false });
    data.append(FP16x16 { mag: 53565, sign: false });
    data.append(FP16x16 { mag: 1198, sign: false });
    data.append(FP16x16 { mag: 31917, sign: false });
    data.append(FP16x16 { mag: 38005, sign: false });
    data.append(FP16x16 { mag: 22276, sign: false });
    data.append(FP16x16 { mag: 34928, sign: false });
    data.append(FP16x16 { mag: 28767, sign: false });
    data.append(FP16x16 { mag: 18918, sign: false });
    data.append(FP16x16 { mag: 40664, sign: false });
    data.append(FP16x16 { mag: 26252, sign: false });
    data.append(FP16x16 { mag: 26596, sign: false });
    data.append(FP16x16 { mag: 8752, sign: false });
    data.append(FP16x16 { mag: 15994, sign: false });
    data.append(FP16x16 { mag: 483, sign: false });
    data.append(FP16x16 { mag: 27831, sign: false });
    data.append(FP16x16 { mag: 28378, sign: false });
    data.append(FP16x16 { mag: 25924, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
