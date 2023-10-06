use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7929856, sign: true });
    data.append(FP16x16 { mag: 1638400, sign: false });
    data.append(FP16x16 { mag: 3342336, sign: true });
    data.append(FP16x16 { mag: 1966080, sign: false });
    data.append(FP16x16 { mag: 3866624, sign: false });
    data.append(FP16x16 { mag: 3014656, sign: false });
    data.append(FP16x16 { mag: 2228224, sign: false });
    data.append(FP16x16 { mag: 5898240, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
