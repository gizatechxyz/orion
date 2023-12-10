use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5177344, sign: false });
    data.append(FP16x16 { mag: 7340032, sign: true });
    data.append(FP16x16 { mag: 7077888, sign: false });
    data.append(FP16x16 { mag: 2752512, sign: true });
    data.append(FP16x16 { mag: 4784128, sign: false });
    data.append(FP16x16 { mag: 131072, sign: true });
    data.append(FP16x16 { mag: 4194304, sign: true });
    data.append(FP16x16 { mag: 2359296, sign: true });
    data.append(FP16x16 { mag: 4915200, sign: true });
    data.append(FP16x16 { mag: 2555904, sign: false });
    data.append(FP16x16 { mag: 458752, sign: false });
    data.append(FP16x16 { mag: 6881280, sign: false });
    data.append(FP16x16 { mag: 1048576, sign: true });
    data.append(FP16x16 { mag: 6160384, sign: true });
    data.append(FP16x16 { mag: 4587520, sign: true });
    data.append(FP16x16 { mag: 589824, sign: true });
    data.append(FP16x16 { mag: 4259840, sign: true });
    data.append(FP16x16 { mag: 2686976, sign: true });
    data.append(FP16x16 { mag: 7405568, sign: true });
    data.append(FP16x16 { mag: 786432, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
