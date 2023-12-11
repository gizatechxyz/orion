use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(27);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 12936, sign: false });
    data.append(FP16x16 { mag: 15752, sign: false });
    data.append(FP16x16 { mag: 17232, sign: false });
    data.append(FP16x16 { mag: 23840, sign: false });
    data.append(FP16x16 { mag: 25056, sign: false });
    data.append(FP16x16 { mag: 27312, sign: false });
    data.append(FP16x16 { mag: 33120, sign: false });
    data.append(FP16x16 { mag: 36128, sign: false });
    data.append(FP16x16 { mag: 43456, sign: false });
    data.append(FP16x16 { mag: 48000, sign: false });
    data.append(FP16x16 { mag: 51680, sign: false });
    data.append(FP16x16 { mag: 56896, sign: false });
    data.append(FP16x16 { mag: 62944, sign: false });
    data.append(FP16x16 { mag: 66496, sign: false });
    data.append(FP16x16 { mag: 70528, sign: false });
    data.append(FP16x16 { mag: 81728, sign: false });
    data.append(FP16x16 { mag: 94976, sign: false });
    data.append(FP16x16 { mag: 103872, sign: false });
    data.append(FP16x16 { mag: 105280, sign: false });
    data.append(FP16x16 { mag: 112128, sign: false });
    data.append(FP16x16 { mag: 138112, sign: false });
    data.append(FP16x16 { mag: 164096, sign: false });
    data.append(FP16x16 { mag: 168192, sign: false });
    data.append(FP16x16 { mag: 168448, sign: false });
    data.append(FP16x16 { mag: 181120, sign: false });
    data.append(FP16x16 { mag: 187776, sign: false });
    data.append(FP16x16 { mag: 195840, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
