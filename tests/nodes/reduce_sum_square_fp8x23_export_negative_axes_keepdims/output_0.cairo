use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 462, sign: false });
    data.append(FP8x23 { mag: 525, sign: false });
    data.append(FP8x23 { mag: 594, sign: false });
    data.append(FP8x23 { mag: 669, sign: false });
    data.append(FP8x23 { mag: 750, sign: false });
    data.append(FP8x23 { mag: 837, sign: false });
    data.append(FP8x23 { mag: 930, sign: false });
    data.append(FP8x23 { mag: 1029, sign: false });
    data.append(FP8x23 { mag: 1134, sign: false });
    TensorTrait::new(shape.span(), data.span())
}