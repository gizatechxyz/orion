use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(5);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 524288, sign: false });
    data.append(FP16x16 { mag: 1376256, sign: false });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 1769472, sign: false });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 2490368, sign: false });
    data.append(FP16x16 { mag: 4325376, sign: false });
    data.append(FP16x16 { mag: 4521984, sign: false });
    data.append(FP16x16 { mag: 4718592, sign: false });
    data.append(FP16x16 { mag: 3538944, sign: false });
    data.append(FP16x16 { mag: 4456448, sign: false });
    data.append(FP16x16 { mag: 7274496, sign: false });
    data.append(FP16x16 { mag: 7471104, sign: false });
    data.append(FP16x16 { mag: 7667712, sign: false });
    data.append(FP16x16 { mag: 5505024, sign: false });
    data.append(FP16x16 { mag: 6422528, sign: false });
    data.append(FP16x16 { mag: 10223616, sign: false });
    data.append(FP16x16 { mag: 10420224, sign: false });
    data.append(FP16x16 { mag: 10616832, sign: false });
    data.append(FP16x16 { mag: 7471104, sign: false });
    data.append(FP16x16 { mag: 8388608, sign: false });
    data.append(FP16x16 { mag: 13172736, sign: false });
    data.append(FP16x16 { mag: 13369344, sign: false });
    data.append(FP16x16 { mag: 13565952, sign: false });
    data.append(FP16x16 { mag: 9437184, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
