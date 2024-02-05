use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Array<Tensor<FP16x16>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1507328, sign: true });
    data.append(FP16x16 { mag: 1966080, sign: true });
    data.append(FP16x16 { mag: 7929856, sign: false });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 4521984, sign: false });
    data.append(FP16x16 { mag: 3866624, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 4521984, sign: true });
    data.append(FP16x16 { mag: 5505024, sign: false });
    data.append(FP16x16 { mag: 4849664, sign: false });
    data.append(FP16x16 { mag: 6815744, sign: false });
    data.append(FP16x16 { mag: 5636096, sign: false });
    data.append(FP16x16 { mag: 7602176, sign: true });
    data.append(FP16x16 { mag: 7208960, sign: true });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 4390912, sign: true });
    data.append(FP16x16 { mag: 6291456, sign: false });
    data.append(FP16x16 { mag: 2818048, sign: true });
    data.append(FP16x16 { mag: 2949120, sign: true });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
