use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 780140544, sign: true });
    data.append(FP8x23 { mag: 209715200, sign: true });
    data.append(FP8x23 { mag: 218103808, sign: true });
    data.append(FP8x23 { mag: 822083584, sign: false });
    data.append(FP8x23 { mag: 478150656, sign: false });
    data.append(FP8x23 { mag: 905969664, sign: true });
    data.append(FP8x23 { mag: 1065353216, sign: true });
    data.append(FP8x23 { mag: 562036736, sign: false });
    data.append(FP8x23 { mag: 620756992, sign: false });
    data.append(FP8x23 { mag: 452984832, sign: true });
    data.append(FP8x23 { mag: 813694976, sign: true });
    data.append(FP8x23 { mag: 109051904, sign: true });
    data.append(FP8x23 { mag: 1031798784, sign: false });
    data.append(FP8x23 { mag: 788529152, sign: false });
    data.append(FP8x23 { mag: 318767104, sign: true });
    data.append(FP8x23 { mag: 377487360, sign: true });
    data.append(FP8x23 { mag: 956301312, sign: false });
    data.append(FP8x23 { mag: 1065353216, sign: true });
    data.append(FP8x23 { mag: 117440512, sign: true });
    data.append(FP8x23 { mag: 931135488, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
