use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 83886080, sign: true });
    data.append(FP8x23 { mag: 167772160, sign: false });
    data.append(FP8x23 { mag: 83886080, sign: true });
    data.append(FP8x23 { mag: 83886080, sign: true });
    data.append(FP8x23 { mag: 167772160, sign: false });
    data.append(FP8x23 { mag: 167772160, sign: false });
    data.append(FP8x23 { mag: 83886080, sign: true });
    data.append(FP8x23 { mag: 83886080, sign: true });
    TensorTrait::new(shape.span(), data.span())
}