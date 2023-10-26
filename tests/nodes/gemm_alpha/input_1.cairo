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
    data.append(FP16x16 { mag: 7870, sign: false });
    data.append(FP16x16 { mag: 11258, sign: false });
    data.append(FP16x16 { mag: 34213, sign: false });
    data.append(FP16x16 { mag: 31148, sign: false });
    data.append(FP16x16 { mag: 29977, sign: false });
    data.append(FP16x16 { mag: 56430, sign: false });
    data.append(FP16x16 { mag: 43116, sign: false });
    data.append(FP16x16 { mag: 22990, sign: false });
    data.append(FP16x16 { mag: 3089, sign: false });
    data.append(FP16x16 { mag: 47936, sign: false });
    data.append(FP16x16 { mag: 13186, sign: false });
    data.append(FP16x16 { mag: 14386, sign: false });
    data.append(FP16x16 { mag: 63802, sign: false });
    data.append(FP16x16 { mag: 19313, sign: false });
    data.append(FP16x16 { mag: 40436, sign: false });
    data.append(FP16x16 { mag: 31890, sign: false });
    data.append(FP16x16 { mag: 34370, sign: false });
    data.append(FP16x16 { mag: 8853, sign: false });
    data.append(FP16x16 { mag: 59520, sign: false });
    data.append(FP16x16 { mag: 40977, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
