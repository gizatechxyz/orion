use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 9215828, sign: false });
    data.append(FP8x23 { mag: 16323477, sign: false });
    data.append(FP8x23 { mag: 20115004, sign: false });
    data.append(FP8x23 { mag: 22716772, sign: false });
    data.append(FP8x23 { mag: 24699744, sign: false });
    data.append(FP8x23 { mag: 26302432, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
