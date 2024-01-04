use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 42564, sign: false });
    data.append(FP16x16 { mag: 18018, sign: false });
    data.append(FP16x16 { mag: 28175, sign: false });
    data.append(FP16x16 { mag: 36784, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
