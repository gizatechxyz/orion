use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);
    shape.append(1);
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 42889, sign: false });
    data.append(FP16x16 { mag: 57394, sign: false });
    data.append(FP16x16 { mag: 119416, sign: true });
    data.append(FP16x16 { mag: 13971, sign: true });
    data.append(FP16x16 { mag: 23832, sign: true });
    data.append(FP16x16 { mag: 128044, sign: true });
    data.append(FP16x16 { mag: 26022, sign: true });
    data.append(FP16x16 { mag: 18751, sign: true });
    data.append(FP16x16 { mag: 151881, sign: true });
    data.append(FP16x16 { mag: 132381, sign: true });
    data.append(FP16x16 { mag: 44651, sign: false });
    data.append(FP16x16 { mag: 48700, sign: true });
    data.append(FP16x16 { mag: 34857, sign: true });
    data.append(FP16x16 { mag: 6509, sign: true });
    data.append(FP16x16 { mag: 27184, sign: false });
    data.append(FP16x16 { mag: 160813, sign: true });
    data.append(FP16x16 { mag: 60755, sign: false });
    data.append(FP16x16 { mag: 70740, sign: true });
    data.append(FP16x16 { mag: 14223, sign: false });
    data.append(FP16x16 { mag: 38701, sign: true });
    data.append(FP16x16 { mag: 186260, sign: true });
    data.append(FP16x16 { mag: 10575, sign: false });
    data.append(FP16x16 { mag: 84458, sign: true });
    data.append(FP16x16 { mag: 48464, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
