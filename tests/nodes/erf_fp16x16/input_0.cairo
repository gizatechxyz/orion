use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7864, sign: false });
    data.append(FP16x16 { mag: 108789, sign: true });
    data.append(FP16x16 { mag: 222822, sign: false });
    data.append(FP16x16 { mag: 314572, sign: false });
    data.append(FP16x16 { mag: 176947, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
