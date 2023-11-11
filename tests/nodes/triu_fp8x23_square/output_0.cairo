use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 377487360, sign: true });
    data.append(FP8x23 { mag: 603979776, sign: true });
    data.append(FP8x23 { mag: 905969664, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 251658240, sign: true });
    data.append(FP8x23 { mag: 897581056, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 150994944, sign: true });
    data.append(FP8x23 { mag: 696254464, sign: true });
    data.append(FP8x23 { mag: 612368384, sign: true });
    data.append(FP8x23 { mag: 847249408, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 41943040, sign: true });
    data.append(FP8x23 { mag: 822083584, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 1040187392, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
