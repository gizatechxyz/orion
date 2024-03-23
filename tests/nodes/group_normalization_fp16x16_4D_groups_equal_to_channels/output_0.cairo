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
    data.append(FP16x16 { mag: 18782, sign: true });
    data.append(FP16x16 { mag: 191298, sign: false });
    data.append(FP16x16 { mag: 50863, sign: false });
    data.append(FP16x16 { mag: 163016, sign: false });
    data.append(FP16x16 { mag: 48612, sign: true });
    data.append(FP16x16 { mag: 48467, sign: true });
    data.append(FP16x16 { mag: 44581, sign: true });
    data.append(FP16x16 { mag: 44437, sign: true });
    data.append(FP16x16 { mag: 41986, sign: false });
    data.append(FP16x16 { mag: 36265, sign: false });
    data.append(FP16x16 { mag: 96198, sign: false });
    data.append(FP16x16 { mag: 211945, sign: false });
    data.append(FP16x16 { mag: 43965, sign: true });
    data.append(FP16x16 { mag: 46579, sign: true });
    data.append(FP16x16 { mag: 49657, sign: true });
    data.append(FP16x16 { mag: 45896, sign: true });
    data.append(FP16x16 { mag: 115480, sign: false });
    data.append(FP16x16 { mag: 231612, sign: false });
    data.append(FP16x16 { mag: 29744, sign: true });
    data.append(FP16x16 { mag: 69047, sign: false });
    data.append(FP16x16 { mag: 46866, sign: true });
    data.append(FP16x16 { mag: 43423, sign: true });
    data.append(FP16x16 { mag: 49217, sign: true });
    data.append(FP16x16 { mag: 46590, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
