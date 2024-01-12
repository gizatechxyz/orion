use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 71268, sign: false });
    data.append(FP16x16 { mag: 7868, sign: false });
    data.append(FP16x16 { mag: 95401, sign: true });
    data.append(FP16x16 { mag: 1087, sign: true });
    data.append(FP16x16 { mag: 1166, sign: false });
    data.append(FP16x16 { mag: 10185, sign: false });
    data.append(FP16x16 { mag: 52837, sign: true });
    data.append(FP16x16 { mag: 5760, sign: true });
    data.append(FP16x16 { mag: 21502, sign: true });
    data.append(FP16x16 { mag: 44185, sign: true });
    data.append(FP16x16 { mag: 39539, sign: false });
    data.append(FP16x16 { mag: 113293, sign: false });
    data.append(FP16x16 { mag: 24873, sign: false });
    data.append(FP16x16 { mag: 124246, sign: false });
    data.append(FP16x16 { mag: 20310, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
