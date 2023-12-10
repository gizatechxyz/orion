use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 10485760, sign: false });
    data.append(FP16x16 { mag: 14942208, sign: false });
    data.append(FP16x16 { mag: 4456448, sign: false });
    data.append(FP16x16 { mag: 10944512, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
