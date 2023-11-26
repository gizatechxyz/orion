use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 112048, sign: false });
    data.append(FP16x16 { mag: 56267, sign: false });
    data.append(FP16x16 { mag: 32529, sign: false });
    data.append(FP16x16 { mag: 22674, sign: false });
    data.append(FP16x16 { mag: 16747, sign: false });
    data.append(FP16x16 { mag: 71161, sign: false });
    data.append(FP16x16 { mag: 152698, sign: false });
    data.append(FP16x16 { mag: 148378, sign: false });
    data.append(FP16x16 { mag: 41538, sign: false });
    data.append(FP16x16 { mag: 42265, sign: false });
    data.append(FP16x16 { mag: 106081, sign: false });
    data.append(FP16x16 { mag: 173882, sign: false });
    data.append(FP16x16 { mag: 6105, sign: false });
    data.append(FP16x16 { mag: 92183, sign: false });
    data.append(FP16x16 { mag: 24661, sign: false });
    data.append(FP16x16 { mag: 131239, sign: false });
    data.append(FP16x16 { mag: 36055, sign: false });
    data.append(FP16x16 { mag: 80890, sign: false });
    data.append(FP16x16 { mag: 51247, sign: false });
    data.append(FP16x16 { mag: 5694, sign: false });
    data.append(FP16x16 { mag: 58318, sign: false });
    data.append(FP16x16 { mag: 64811, sign: false });
    data.append(FP16x16 { mag: 192104, sign: false });
    data.append(FP16x16 { mag: 97136, sign: false });
    data.append(FP16x16 { mag: 94797, sign: false });
    data.append(FP16x16 { mag: 173197, sign: false });
    data.append(FP16x16 { mag: 137788, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
