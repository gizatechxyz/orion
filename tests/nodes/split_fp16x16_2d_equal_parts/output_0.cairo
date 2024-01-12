use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Array<Tensor<FP16x16>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7274496, sign: true });
    data.append(FP16x16 { mag: 1507328, sign: true });
    data.append(FP16x16 { mag: 3604480, sign: false });
    data.append(FP16x16 { mag: 917504, sign: true });
    data.append(FP16x16 { mag: 3538944, sign: true });
    data.append(FP16x16 { mag: 1835008, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7864320, sign: true });
    data.append(FP16x16 { mag: 786432, sign: false });
    data.append(FP16x16 { mag: 6422528, sign: false });
    data.append(FP16x16 { mag: 3407872, sign: false });
    data.append(FP16x16 { mag: 3145728, sign: false });
    data.append(FP16x16 { mag: 8257536, sign: true });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
