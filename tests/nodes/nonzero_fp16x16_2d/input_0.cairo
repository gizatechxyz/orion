use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 4980736, sign: false });
    data.append(FP16x16 { mag: 458752, sign: false });
    data.append(FP16x16 { mag: 6815744, sign: false });
    data.append(FP16x16 { mag: 4259840, sign: true });
    data.append(FP16x16 { mag: 5046272, sign: false });
    data.append(FP16x16 { mag: 2555904, sign: true });
    data.append(FP16x16 { mag: 2031616, sign: false });
    data.append(FP16x16 { mag: 2555904, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
