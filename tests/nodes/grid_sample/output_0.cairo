use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(6);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 9830, sign: false });
    data.append(FP16x16 { mag: 36044, sign: false });
    data.append(FP16x16 { mag: 62259, sign: false });
    data.append(FP16x16 { mag: 88473, sign: false });
    data.append(FP16x16 { mag: 49152, sign: false });
    data.append(FP16x16 { mag: 39321, sign: false });
    data.append(FP16x16 { mag: 98303, sign: false });
    data.append(FP16x16 { mag: 150732, sign: false });
    data.append(FP16x16 { mag: 203161, sign: false });
    data.append(FP16x16 { mag: 255590, sign: false });
    data.append(FP16x16 { mag: 137625, sign: false });
    data.append(FP16x16 { mag: 144179, sign: false });
    data.append(FP16x16 { mag: 308019, sign: false });
    data.append(FP16x16 { mag: 360448, sign: false });
    data.append(FP16x16 { mag: 412876, sign: false });
    data.append(FP16x16 { mag: 465305, sign: false });
    data.append(FP16x16 { mag: 242483, sign: false });
    data.append(FP16x16 { mag: 249036, sign: false });
    data.append(FP16x16 { mag: 517734, sign: false });
    data.append(FP16x16 { mag: 570163, sign: false });
    data.append(FP16x16 { mag: 622592, sign: false });
    data.append(FP16x16 { mag: 675020, sign: false });
    data.append(FP16x16 { mag: 347340, sign: false });
    data.append(FP16x16 { mag: 353894, sign: false });
    data.append(FP16x16 { mag: 727449, sign: false });
    data.append(FP16x16 { mag: 779878, sign: false });
    data.append(FP16x16 { mag: 832307, sign: false });
    data.append(FP16x16 { mag: 884736, sign: false });
    data.append(FP16x16 { mag: 452198, sign: false });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 403046, sign: false });
    data.append(FP16x16 { mag: 429260, sign: false });
    data.append(FP16x16 { mag: 455475, sign: false });
    data.append(FP16x16 { mag: 481689, sign: false });
    data.append(FP16x16 { mag: 245760, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
