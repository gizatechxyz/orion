use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 10671, sign: false });
    data.append(FP16x16 { mag: 42014, sign: false });
    data.append(FP16x16 { mag: 54635, sign: false });
    data.append(FP16x16 { mag: 20143, sign: false });
    data.append(FP16x16 { mag: 23206, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
