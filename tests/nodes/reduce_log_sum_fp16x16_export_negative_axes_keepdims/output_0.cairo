use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 15030367, sign: false });
    data.append(FP8x23 { mag: 17443619, sign: false });
    data.append(FP8x23 { mag: 19315483, sign: false });
    data.append(FP8x23 { mag: 20844907, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
