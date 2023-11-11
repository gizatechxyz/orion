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
    data.append(FP16x16 { mag: 2949120, sign: false });
    data.append(FP16x16 { mag: 4128768, sign: false });
    data.append(FP16x16 { mag: 7012352, sign: false });
    data.append(FP16x16 { mag: 1441792, sign: false });
    data.append(FP16x16 { mag: 5046272, sign: true });
    data.append(FP16x16 { mag: 1245184, sign: false });
    data.append(FP16x16 { mag: 2490368, sign: false });
    data.append(FP16x16 { mag: 983040, sign: false });
    data.append(FP16x16 { mag: 2949120, sign: true });
    data.append(FP16x16 { mag: 5701632, sign: false });
    data.append(FP16x16 { mag: 393216, sign: true });
    data.append(FP16x16 { mag: 6488064, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 5111808, sign: true });
    data.append(FP16x16 { mag: 6946816, sign: true });
    data.append(FP16x16 { mag: 4259840, sign: true });
    data.append(FP16x16 { mag: 5111808, sign: false });
    data.append(FP16x16 { mag: 5373952, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
