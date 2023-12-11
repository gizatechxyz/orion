use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 150528, sign: true });
    data.append(FP16x16 { mag: 126144, sign: false });
    data.append(FP16x16 { mag: 190208, sign: false });
    data.append(FP16x16 { mag: 41088, sign: true });
    data.append(FP16x16 { mag: 26592, sign: false });
    data.append(FP16x16 { mag: 26352, sign: true });
    data.append(FP16x16 { mag: 145664, sign: false });
    data.append(FP16x16 { mag: 92928, sign: true });
    data.append(FP16x16 { mag: 195200, sign: false });
    data.append(FP16x16 { mag: 77824, sign: false });
    data.append(FP16x16 { mag: 97664, sign: true });
    data.append(FP16x16 { mag: 166144, sign: false });
    data.append(FP16x16 { mag: 42112, sign: true });
    data.append(FP16x16 { mag: 32304, sign: true });
    data.append(FP16x16 { mag: 126464, sign: false });
    data.append(FP16x16 { mag: 110400, sign: false });
    data.append(FP16x16 { mag: 146688, sign: false });
    data.append(FP16x16 { mag: 148608, sign: false });
    data.append(FP16x16 { mag: 126784, sign: false });
    data.append(FP16x16 { mag: 172288, sign: true });
    data.append(FP16x16 { mag: 92480, sign: true });
    data.append(FP16x16 { mag: 185344, sign: true });
    data.append(FP16x16 { mag: 162688, sign: false });
    data.append(FP16x16 { mag: 125312, sign: true });
    data.append(FP16x16 { mag: 25024, sign: true });
    data.append(FP16x16 { mag: 108800, sign: true });
    data.append(FP16x16 { mag: 194304, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
