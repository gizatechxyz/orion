use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 15751, sign: false });
    data.append(FP16x16 { mag: 62952, sign: false });
    data.append(FP16x16 { mag: 181085, sign: false });
    data.append(FP16x16 { mag: 56888, sign: false });
    data.append(FP16x16 { mag: 187741, sign: false });
    data.append(FP16x16 { mag: 43455, sign: false });
    data.append(FP16x16 { mag: 33132, sign: false });
    data.append(FP16x16 { mag: 48004, sign: false });
    data.append(FP16x16 { mag: 195778, sign: false });
    data.append(FP16x16 { mag: 25051, sign: false });
    data.append(FP16x16 { mag: 138080, sign: false });
    data.append(FP16x16 { mag: 94981, sign: false });
    data.append(FP16x16 { mag: 103864, sign: false });
    data.append(FP16x16 { mag: 70552, sign: false });
    data.append(FP16x16 { mag: 168132, sign: false });
    data.append(FP16x16 { mag: 168416, sign: false });
    data.append(FP16x16 { mag: 164041, sign: false });
    data.append(FP16x16 { mag: 36136, sign: false });
    data.append(FP16x16 { mag: 51675, sign: false });
    data.append(FP16x16 { mag: 105268, sign: false });
    data.append(FP16x16 { mag: 12938, sign: false });
    data.append(FP16x16 { mag: 112152, sign: false });
    data.append(FP16x16 { mag: 23840, sign: false });
    data.append(FP16x16 { mag: 66491, sign: false });
    data.append(FP16x16 { mag: 27314, sign: false });
    data.append(FP16x16 { mag: 81759, sign: false });
    data.append(FP16x16 { mag: 17239, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
