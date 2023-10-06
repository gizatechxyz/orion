use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 427819008, sign: false });
    data.append(FP8x23 { mag: 427819008, sign: true });
    data.append(FP8x23 { mag: 385875968, sign: false });
    data.append(FP8x23 { mag: 905969664, sign: true });
    data.append(FP8x23 { mag: 360710144, sign: true });
    data.append(FP8x23 { mag: 897581056, sign: false });
    data.append(FP8x23 { mag: 58720256, sign: false });
    data.append(FP8x23 { mag: 838860800, sign: true });
    TensorTrait::new(shape.span(), data.span())
}