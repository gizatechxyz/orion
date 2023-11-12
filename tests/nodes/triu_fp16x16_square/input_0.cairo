use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 3735552, sign: true });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 5767168, sign: true });
    data.append(FP16x16 { mag: 8192000, sign: true });
    data.append(FP16x16 { mag: 6422528, sign: true });
    data.append(FP16x16 { mag: 3276800, sign: false });
    data.append(FP16x16 { mag: 5111808, sign: true });
    data.append(FP16x16 { mag: 7536640, sign: false });
    data.append(FP16x16 { mag: 5177344, sign: true });
    data.append(FP16x16 { mag: 2097152, sign: true });
    data.append(FP16x16 { mag: 786432, sign: true });
    data.append(FP16x16 { mag: 3801088, sign: true });
    data.append(FP16x16 { mag: 5767168, sign: true });
    data.append(FP16x16 { mag: 5570560, sign: true });
    data.append(FP16x16 { mag: 589824, sign: true });
    data.append(FP16x16 { mag: 6553600, sign: true });
    data.append(FP16x16 { mag: 6488064, sign: false });
    data.append(FP16x16 { mag: 4259840, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
