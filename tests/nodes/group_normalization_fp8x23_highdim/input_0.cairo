use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);
    shape.append(1);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3040733, sign: true });
    data.append(FP8x23 { mag: 10242773, sign: true });
    data.append(FP8x23 { mag: 5069687, sign: true });
    data.append(FP8x23 { mag: 2238636, sign: true });
    data.append(FP8x23 { mag: 13113365, sign: false });
    data.append(FP8x23 { mag: 7513804, sign: false });
    data.append(FP8x23 { mag: 8561328, sign: true });
    data.append(FP8x23 { mag: 1114614, sign: true });
    data.append(FP8x23 { mag: 4873273, sign: true });
    data.append(FP8x23 { mag: 19895732, sign: true });
    data.append(FP8x23 { mag: 1629211, sign: true });
    data.append(FP8x23 { mag: 4649699, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
