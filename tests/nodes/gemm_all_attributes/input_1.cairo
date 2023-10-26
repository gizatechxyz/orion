use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 951, sign: false });
    data.append(FP16x16 { mag: 60848, sign: false });
    data.append(FP16x16 { mag: 51199, sign: false });
    data.append(FP16x16 { mag: 16691, sign: false });
    data.append(FP16x16 { mag: 14621, sign: false });
    data.append(FP16x16 { mag: 51626, sign: false });
    data.append(FP16x16 { mag: 33242, sign: false });
    data.append(FP16x16 { mag: 36152, sign: false });
    data.append(FP16x16 { mag: 41495, sign: false });
    data.append(FP16x16 { mag: 21214, sign: false });
    data.append(FP16x16 { mag: 63748, sign: false });
    data.append(FP16x16 { mag: 9058, sign: false });
    data.append(FP16x16 { mag: 38129, sign: false });
    data.append(FP16x16 { mag: 32448, sign: false });
    data.append(FP16x16 { mag: 34299, sign: false });
    data.append(FP16x16 { mag: 28592, sign: false });
    data.append(FP16x16 { mag: 60878, sign: false });
    data.append(FP16x16 { mag: 1143, sign: false });
    data.append(FP16x16 { mag: 2602, sign: false });
    data.append(FP16x16 { mag: 12136, sign: false });
    TensorTrait::new(shape.span(), data.span())
}