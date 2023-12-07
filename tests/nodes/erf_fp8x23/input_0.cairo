use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1006632, sign: false });
    data.append(FP8x23 { mag: 13925089, sign: true });
    data.append(FP8x23 { mag: 28521267, sign: false });
    data.append(FP8x23 { mag: 40265318, sign: false });
    data.append(FP8x23 { mag: 22649241, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
