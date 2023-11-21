use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 12520, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 107103, sign: true });
    data.append(FP16x16 { mag: 13061, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 106120, sign: false });
    data.append(FP16x16 { mag: 91999, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 31472, sign: false });
    data.append(FP16x16 { mag: 58223, sign: true });
    data.append(FP16x16 { mag: 124731, sign: false });
    data.append(FP16x16 { mag: 41827, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 3006, sign: false });
    data.append(FP16x16 { mag: 51142, sign: false });
    data.append(FP16x16 { mag: 20061, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 34238, sign: false });
    data.append(FP16x16 { mag: 107948, sign: false });
    data.append(FP16x16 { mag: 61481, sign: true });
    data.append(FP16x16 { mag: 18160, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 14890, sign: false });
    data.append(FP16x16 { mag: 122450, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
