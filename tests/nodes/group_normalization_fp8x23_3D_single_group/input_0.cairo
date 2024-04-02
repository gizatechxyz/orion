use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 11527135, sign: false });
    data.append(FP8x23 { mag: 7317171, sign: false });
    data.append(FP8x23 { mag: 4837434, sign: true });
    data.append(FP8x23 { mag: 12308917, sign: true });
    data.append(FP8x23 { mag: 6337801, sign: true });
    data.append(FP8x23 { mag: 1555501, sign: false });
    data.append(FP8x23 { mag: 4103700, sign: true });
    data.append(FP8x23 { mag: 2660544, sign: true });
    data.append(FP8x23 { mag: 16132761, sign: true });
    data.append(FP8x23 { mag: 1764195, sign: true });
    data.append(FP8x23 { mag: 4730136, sign: false });
    data.append(FP8x23 { mag: 7734009, sign: false });
    data.append(FP8x23 { mag: 4765283, sign: true });
    data.append(FP8x23 { mag: 6610290, sign: false });
    data.append(FP8x23 { mag: 12941895, sign: false });
    data.append(FP8x23 { mag: 2841994, sign: false });
    data.append(FP8x23 { mag: 4470684, sign: true });
    data.append(FP8x23 { mag: 13571680, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
