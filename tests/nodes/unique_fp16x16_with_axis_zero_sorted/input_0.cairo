use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 87962, sign: true });
    data.append(FP16x16 { mag: 9323, sign: false });
    data.append(FP16x16 { mag: 66240, sign: true });
    data.append(FP16x16 { mag: 156291, sign: false });
    data.append(FP16x16 { mag: 78003, sign: false });
    data.append(FP16x16 { mag: 184061, sign: false });
    data.append(FP16x16 { mag: 28971, sign: false });
    data.append(FP16x16 { mag: 47164, sign: true });
    data.append(FP16x16 { mag: 143808, sign: false });
    data.append(FP16x16 { mag: 17239, sign: false });
    data.append(FP16x16 { mag: 168224, sign: false });
    data.append(FP16x16 { mag: 140637, sign: false });
    data.append(FP16x16 { mag: 196341, sign: false });
    data.append(FP16x16 { mag: 126336, sign: false });
    data.append(FP16x16 { mag: 190457, sign: true });
    data.append(FP16x16 { mag: 196122, sign: false });
    data.append(FP16x16 { mag: 129902, sign: true });
    data.append(FP16x16 { mag: 47503, sign: true });
    data.append(FP16x16 { mag: 173883, sign: true });
    data.append(FP16x16 { mag: 179343, sign: false });
    data.append(FP16x16 { mag: 151895, sign: true });
    data.append(FP16x16 { mag: 195200, sign: false });
    data.append(FP16x16 { mag: 71299, sign: false });
    data.append(FP16x16 { mag: 71498, sign: false });
    data.append(FP16x16 { mag: 98908, sign: false });
    data.append(FP16x16 { mag: 18762, sign: true });
    data.append(FP16x16 { mag: 194071, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
