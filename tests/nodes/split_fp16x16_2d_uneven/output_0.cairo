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
    data.append(FP16x16 { mag: 4980736, sign: false });
    data.append(FP16x16 { mag: 6750208, sign: true });
    data.append(FP16x16 { mag: 6488064, sign: true });
    data.append(FP16x16 { mag: 4849664, sign: false });
    data.append(FP16x16 { mag: 3538944, sign: true });
    data.append(FP16x16 { mag: 7077888, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2490368, sign: false });
    data.append(FP16x16 { mag: 1245184, sign: true });
    data.append(FP16x16 { mag: 1310720, sign: false });
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 8192000, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2686976, sign: true });
    data.append(FP16x16 { mag: 3801088, sign: false });
    data.append(FP16x16 { mag: 1441792, sign: true });
    data.append(FP16x16 { mag: 6553600, sign: true });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
