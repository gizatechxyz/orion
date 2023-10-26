use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 17180, sign: false });
    data.append(FP16x16 { mag: 16229, sign: false });
    data.append(FP16x16 { mag: 15872, sign: false });
    data.append(FP16x16 { mag: 41908, sign: false });
    data.append(FP16x16 { mag: 19343, sign: false });
    data.append(FP16x16 { mag: 23171, sign: false });
    data.append(FP16x16 { mag: 40127, sign: false });
    data.append(FP16x16 { mag: 21249, sign: false });
    data.append(FP16x16 { mag: 14046, sign: false });
    data.append(FP16x16 { mag: 17154, sign: false });
    data.append(FP16x16 { mag: 31592, sign: false });
    data.append(FP16x16 { mag: 38555, sign: false });
    TensorTrait::new(shape.span(), data.span())
}