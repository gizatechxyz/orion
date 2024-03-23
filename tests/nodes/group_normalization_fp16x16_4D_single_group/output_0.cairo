use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 101160, sign: true });
    data.append(FP16x16 { mag: 9485, sign: false });
    data.append(FP16x16 { mag: 159168, sign: false });
    data.append(FP16x16 { mag: 44602, sign: false });
    data.append(FP16x16 { mag: 170299, sign: true });
    data.append(FP16x16 { mag: 109114, sign: false });
    data.append(FP16x16 { mag: 66188, sign: true });
    data.append(FP16x16 { mag: 35122, sign: false });
    data.append(FP16x16 { mag: 82485, sign: true });
    data.append(FP16x16 { mag: 20048, sign: true });
    data.append(FP16x16 { mag: 53889, sign: false });
    data.append(FP16x16 { mag: 128356, sign: false });
    data.append(FP16x16 { mag: 45805, sign: true });
    data.append(FP16x16 { mag: 130077, sign: false });
    data.append(FP16x16 { mag: 205922, sign: true });
    data.append(FP16x16 { mag: 12862, sign: true });
    data.append(FP16x16 { mag: 216815, sign: false });
    data.append(FP16x16 { mag: 28526, sign: true });
    data.append(FP16x16 { mag: 34016, sign: true });
    data.append(FP16x16 { mag: 81362, sign: false });
    data.append(FP16x16 { mag: 159512, sign: false });
    data.append(FP16x16 { mag: 16281, sign: true });
    data.append(FP16x16 { mag: 67813, sign: true });
    data.append(FP16x16 { mag: 6445, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
