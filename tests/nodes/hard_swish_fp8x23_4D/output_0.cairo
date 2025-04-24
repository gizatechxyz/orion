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
    data.append(FP8x23 { mag: 11899102, sign: false });
    data.append(FP8x23 { mag: 8981221, sign: false });
    data.append(FP8x23 { mag: 674092, sign: false });
    data.append(FP8x23 { mag: 3137902, sign: true });
    data.append(FP8x23 { mag: 835292, sign: false });
    data.append(FP8x23 { mag: 7316199, sign: false });
    data.append(FP8x23 { mag: 2586737, sign: true });
    data.append(FP8x23 { mag: 8536633, sign: false });
    data.append(FP8x23 { mag: 3357570, sign: false });
    data.append(FP8x23 { mag: 1795812, sign: true });
    data.append(FP8x23 { mag: 2559583, sign: true });
    data.append(FP8x23 { mag: 153225, sign: true });
    data.append(FP8x23 { mag: 3836591, sign: false });
    data.append(FP8x23 { mag: 1014581, sign: true });
    data.append(FP8x23 { mag: 2797336, sign: true });
    data.append(FP8x23 { mag: 8330698, sign: false });
    data.append(FP8x23 { mag: 3139558, sign: true });
    data.append(FP8x23 { mag: 19731334, sign: false });
    data.append(FP8x23 { mag: 2436528, sign: true });
    data.append(FP8x23 { mag: 2364559, sign: true });
    data.append(FP8x23 { mag: 935970, sign: false });
    data.append(FP8x23 { mag: 9323093, sign: false });
    data.append(FP8x23 { mag: 1974152, sign: true });
    data.append(FP8x23 { mag: 14598985, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
