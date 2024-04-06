use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 14934920, sign: false });
    data.append(FP8x23 { mag: 12122742, sign: false });
    data.append(FP8x23 { mag: 1282795, sign: false });
    data.append(FP8x23 { mag: 13210522, sign: true });
    data.append(FP8x23 { mag: 1572345, sign: false });
    data.append(FP8x23 { mag: 10364115, sign: false });
    data.append(FP8x23 { mag: 7278677, sign: true });
    data.append(FP8x23 { mag: 11665645, sign: false });
    data.append(FP8x23 { mag: 5509122, sign: false });
    data.append(FP8x23 { mag: 20825694, sign: true });
    data.append(FP8x23 { mag: 18014452, sign: true });
    data.append(FP8x23 { mag: 310275, sign: true });
    data.append(FP8x23 { mag: 6163598, sign: false });
    data.append(FP8x23 { mag: 2226072, sign: true });
    data.append(FP8x23 { mag: 16770404, sign: true });
    data.append(FP8x23 { mag: 11450969, sign: false });
    data.append(FP8x23 { mag: 13140147, sign: true });
    data.append(FP8x23 { mag: 21349966, sign: false });
    data.append(FP8x23 { mag: 6608367, sign: true });
    data.append(FP8x23 { mag: 6312547, sign: true });
    data.append(FP8x23 { mag: 1750218, sign: false });
    data.append(FP8x23 { mag: 12468561, sign: false });
    data.append(FP8x23 { mag: 4903895, sign: true });
    data.append(FP8x23 { mag: 17302212, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
