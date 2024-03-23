use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8213730, sign: false });
    data.append(FP8x23 { mag: 13088612, sign: false });
    data.append(FP8x23 { mag: 3519516, sign: true });
    data.append(FP8x23 { mag: 3863764, sign: true });
    data.append(FP8x23 { mag: 9538733, sign: true });
    data.append(FP8x23 { mag: 290689, sign: true });
    data.append(FP8x23 { mag: 544452, sign: false });
    data.append(FP8x23 { mag: 255411, sign: false });
    data.append(FP8x23 { mag: 16147706, sign: true });
    data.append(FP8x23 { mag: 3590977, sign: false });
    data.append(FP8x23 { mag: 8389978, sign: true });
    data.append(FP8x23 { mag: 9643816, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
