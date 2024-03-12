use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 57225710, sign: true });
    data.append(FP8x23 { mag: 206036055, sign: false });
    data.append(FP8x23 { mag: 52167302, sign: false });
    data.append(FP8x23 { mag: 70273911, sign: true });
    data.append(FP8x23 { mag: 116495619, sign: false });
    data.append(FP8x23 { mag: 170231765, sign: true });
    data.append(FP8x23 { mag: 200075176, sign: true });
    data.append(FP8x23 { mag: 51815704, sign: false });
    data.append(FP8x23 { mag: 62353036, sign: true });
    data.append(FP8x23 { mag: 105106308, sign: false });
    data.append(FP8x23 { mag: 228441912, sign: false });
    data.append(FP8x23 { mag: 183099120, sign: true });
    data.append(FP8x23 { mag: 118036599, sign: false });
    data.append(FP8x23 { mag: 137653027, sign: false });
    data.append(FP8x23 { mag: 232603284, sign: false });
    data.append(FP8x23 { mag: 32598207, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
