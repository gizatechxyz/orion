use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 75497472, sign: false });
    data.append(FP8x23 { mag: 494927872, sign: true });
    data.append(FP8x23 { mag: 75497472, sign: true });
    data.append(FP8x23 { mag: 1056964608, sign: false });
    data.append(FP8x23 { mag: 310378496, sign: false });
    data.append(FP8x23 { mag: 218103808, sign: false });
    data.append(FP8x23 { mag: 1006632960, sign: false });
    data.append(FP8x23 { mag: 427819008, sign: true });
    data.append(FP8x23 { mag: 520093696, sign: true });
    data.append(FP8x23 { mag: 109051904, sign: true });
    data.append(FP8x23 { mag: 612368384, sign: true });
    data.append(FP8x23 { mag: 721420288, sign: false });
    data.append(FP8x23 { mag: 662700032, sign: false });
    data.append(FP8x23 { mag: 553648128, sign: true });
    data.append(FP8x23 { mag: 33554432, sign: false });
    data.append(FP8x23 { mag: 335544320, sign: true });
    data.append(FP8x23 { mag: 981467136, sign: false });
    data.append(FP8x23 { mag: 562036736, sign: true });
    data.append(FP8x23 { mag: 637534208, sign: false });
    data.append(FP8x23 { mag: 109051904, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
