use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 8602, sign: true });
    data.append(FP16x16 { mag: 134121, sign: true });
    data.append(FP16x16 { mag: 39230, sign: false });
    data.append(FP16x16 { mag: 17052, sign: true });
    data.append(FP16x16 { mag: 24886, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
