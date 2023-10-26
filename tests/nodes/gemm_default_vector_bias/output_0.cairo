use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 141552, sign: false });
    data.append(FP16x16 { mag: 182056, sign: false });
    data.append(FP16x16 { mag: 169805, sign: false });
    data.append(FP16x16 { mag: 151525, sign: false });
    data.append(FP16x16 { mag: 138029, sign: false });
    data.append(FP16x16 { mag: 183035, sign: false });
    data.append(FP16x16 { mag: 124434, sign: false });
    data.append(FP16x16 { mag: 101476, sign: false });
    TensorTrait::new(shape.span(), data.span())
}