use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 524288, sign: true });
    data.append(FP16x16 { mag: 1245184, sign: true });
    data.append(FP16x16 { mag: 3801088, sign: false });
    data.append(FP16x16 { mag: 1179648, sign: false });
    data.append(FP16x16 { mag: 4980736, sign: false });
    data.append(FP16x16 { mag: 917504, sign: true });
    data.append(FP16x16 { mag: 3997696, sign: false });
    data.append(FP16x16 { mag: 3538944, sign: false });
    data.append(FP16x16 { mag: 1114112, sign: false });
    data.append(FP16x16 { mag: 2490368, sign: true });
    data.append(FP16x16 { mag: 720896, sign: true });
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 5898240, sign: false });
    data.append(FP16x16 { mag: 6750208, sign: false });
    data.append(FP16x16 { mag: 7995392, sign: false });
    data.append(FP16x16 { mag: 2031616, sign: false });
    data.append(FP16x16 { mag: 5308416, sign: true });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 7929856, sign: false });
    data.append(FP16x16 { mag: 4063232, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
