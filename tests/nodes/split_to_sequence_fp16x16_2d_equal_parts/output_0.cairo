use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Array<Tensor<FP16x16>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2752512, sign: true });
    data.append(FP16x16 { mag: 524288, sign: false });
    data.append(FP16x16 { mag: 5636096, sign: true });
    data.append(FP16x16 { mag: 6684672, sign: true });
    data.append(FP16x16 { mag: 4718592, sign: true });
    data.append(FP16x16 { mag: 4063232, sign: true });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 4718592, sign: true });
    data.append(FP16x16 { mag: 6094848, sign: false });
    data.append(FP16x16 { mag: 3080192, sign: true });
    data.append(FP16x16 { mag: 1310720, sign: false });
    data.append(FP16x16 { mag: 786432, sign: true });
    data.append(FP16x16 { mag: 2686976, sign: true });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
