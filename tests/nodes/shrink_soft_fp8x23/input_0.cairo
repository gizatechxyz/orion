use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 15286254, sign: false });
    data.append(FP8x23 { mag: 8658111, sign: false });
    data.append(FP8x23 { mag: 20563043, sign: true });
    data.append(FP8x23 { mag: 1544545, sign: true });
    data.append(FP8x23 { mag: 17891099, sign: false });
    data.append(FP8x23 { mag: 12585699, sign: true });
    data.append(FP8x23 { mag: 15983949, sign: false });
    data.append(FP8x23 { mag: 2040356, sign: false });
    data.append(FP8x23 { mag: 16325343, sign: false });
    data.append(FP8x23 { mag: 8838001, sign: false });
    data.append(FP8x23 { mag: 12943137, sign: true });
    data.append(FP8x23 { mag: 11093722, sign: false });
    data.append(FP8x23 { mag: 2693368, sign: false });
    data.append(FP8x23 { mag: 20149881, sign: false });
    data.append(FP8x23 { mag: 916303, sign: false });
    data.append(FP8x23 { mag: 23170082, sign: true });
    data.append(FP8x23 { mag: 21298676, sign: false });
    data.append(FP8x23 { mag: 24498513, sign: true });
    data.append(FP8x23 { mag: 10219527, sign: true });
    data.append(FP8x23 { mag: 13912176, sign: false });
    data.append(FP8x23 { mag: 11208649, sign: true });
    data.append(FP8x23 { mag: 17452815, sign: false });
    data.append(FP8x23 { mag: 24243821, sign: true });
    data.append(FP8x23 { mag: 190936, sign: false });
    data.append(FP8x23 { mag: 11213942, sign: false });
    data.append(FP8x23 { mag: 21569767, sign: false });
    data.append(FP8x23 { mag: 10078186, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
