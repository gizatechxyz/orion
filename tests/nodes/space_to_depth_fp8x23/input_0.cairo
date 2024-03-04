use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 19097721, sign: true });
    data.append(FP8x23 { mag: 20388727, sign: false });
    data.append(FP8x23 { mag: 18733446, sign: false });
    data.append(FP8x23 { mag: 5803068, sign: false });
    data.append(FP8x23 { mag: 21193100, sign: false });
    data.append(FP8x23 { mag: 7531714, sign: false });
    data.append(FP8x23 { mag: 16983892, sign: false });
    data.append(FP8x23 { mag: 18182574, sign: true });
    data.append(FP8x23 { mag: 3066595, sign: false });
    data.append(FP8x23 { mag: 17329855, sign: false });
    data.append(FP8x23 { mag: 14812767, sign: true });
    data.append(FP8x23 { mag: 5408423, sign: false });
    data.append(FP8x23 { mag: 23872828, sign: true });
    data.append(FP8x23 { mag: 19363658, sign: false });
    data.append(FP8x23 { mag: 6503203, sign: false });
    data.append(FP8x23 { mag: 6090326, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
