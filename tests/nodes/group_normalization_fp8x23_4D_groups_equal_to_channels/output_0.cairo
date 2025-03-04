use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 5347275, sign: false });
    data.append(FP8x23 { mag: 25530942, sign: false });
    data.append(FP8x23 { mag: 7513289, sign: false });
    data.append(FP8x23 { mag: 15780413, sign: false });
    data.append(FP8x23 { mag: 5654052, sign: true });
    data.append(FP8x23 { mag: 6738215, sign: true });
    data.append(FP8x23 { mag: 6225779, sign: true });
    data.append(FP8x23 { mag: 6015155, sign: true });
    data.append(FP8x23 { mag: 17600448, sign: false });
    data.append(FP8x23 { mag: 6808013, sign: false });
    data.append(FP8x23 { mag: 25138274, sign: false });
    data.append(FP8x23 { mag: 4625185, sign: false });
    data.append(FP8x23 { mag: 6063410, sign: true });
    data.append(FP8x23 { mag: 5834588, sign: true });
    data.append(FP8x23 { mag: 5983876, sign: true });
    data.append(FP8x23 { mag: 6751327, sign: true });
    data.append(FP8x23 { mag: 13924825, sign: false });
    data.append(FP8x23 { mag: 108653, sign: true });
    data.append(FP8x23 { mag: 23256002, sign: false });
    data.append(FP8x23 { mag: 17099748, sign: false });
    data.append(FP8x23 { mag: 6118866, sign: true });
    data.append(FP8x23 { mag: 5841536, sign: true });
    data.append(FP8x23 { mag: 5874278, sign: true });
    data.append(FP8x23 { mag: 6798521, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
