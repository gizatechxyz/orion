use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2621440, sign: false });
    data.append(FP16x16 { mag: 7143424, sign: false });
    data.append(FP16x16 { mag: 1245184, sign: false });
    data.append(FP16x16 { mag: 7471104, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
