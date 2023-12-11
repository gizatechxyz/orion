use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 77831, sign: false });
    data.append(FP16x16 { mag: 97693, sign: true });
    data.append(FP16x16 { mag: 166198, sign: false });
    data.append(FP16x16 { mag: 42115, sign: true });
    data.append(FP16x16 { mag: 32310, sign: true });
    data.append(FP16x16 { mag: 126486, sign: false });
    data.append(FP16x16 { mag: 110391, sign: false });
    data.append(FP16x16 { mag: 146684, sign: false });
    data.append(FP16x16 { mag: 148565, sign: false });
    data.append(FP16x16 { mag: 150552, sign: true });
    data.append(FP16x16 { mag: 126123, sign: false });
    data.append(FP16x16 { mag: 190215, sign: false });
    data.append(FP16x16 { mag: 41080, sign: true });
    data.append(FP16x16 { mag: 26594, sign: false });
    data.append(FP16x16 { mag: 26344, sign: true });
    data.append(FP16x16 { mag: 145709, sign: false });
    data.append(FP16x16 { mag: 92915, sign: true });
    data.append(FP16x16 { mag: 195207, sign: false });
    data.append(FP16x16 { mag: 126804, sign: false });
    data.append(FP16x16 { mag: 172341, sign: true });
    data.append(FP16x16 { mag: 92491, sign: true });
    data.append(FP16x16 { mag: 185375, sign: true });
    data.append(FP16x16 { mag: 162645, sign: false });
    data.append(FP16x16 { mag: 125282, sign: true });
    data.append(FP16x16 { mag: 25029, sign: true });
    data.append(FP16x16 { mag: 108797, sign: true });
    data.append(FP16x16 { mag: 194338, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
