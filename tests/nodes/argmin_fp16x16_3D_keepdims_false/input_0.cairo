use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7929856, sign: true });
    data.append(FP16x16 { mag: 2228224, sign: true });
    data.append(FP16x16 { mag: 851968, sign: false });
    data.append(FP16x16 { mag: 3080192, sign: false });
    data.append(FP16x16 { mag: 4915200, sign: true });
    data.append(FP16x16 { mag: 1638400, sign: true });
    data.append(FP16x16 { mag: 6815744, sign: true });
    data.append(FP16x16 { mag: 2359296, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
