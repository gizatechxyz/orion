use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 989855744, sign: false });
    data.append(FP8x23 { mag: 536870912, sign: true });
    data.append(FP8x23 { mag: 360710144, sign: false });
    data.append(FP8x23 { mag: 780140544, sign: true });
    data.append(FP8x23 { mag: 58720256, sign: true });
    data.append(FP8x23 { mag: 301989888, sign: false });
    data.append(FP8x23 { mag: 889192448, sign: true });
    data.append(FP8x23 { mag: 562036736, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
