use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 721420288, sign: true });
    data.append(FP8x23 { mag: 335544320, sign: false });
    data.append(FP8x23 { mag: 796917760, sign: true });
    data.append(FP8x23 { mag: 75497472, sign: false });
    data.append(FP8x23 { mag: 33554432, sign: true });
    data.append(FP8x23 { mag: 1048576000, sign: true });
    data.append(FP8x23 { mag: 914358272, sign: false });
    data.append(FP8x23 { mag: 427819008, sign: false });
    data.append(FP8x23 { mag: 511705088, sign: true });
    data.append(FP8x23 { mag: 612368384, sign: false });
    data.append(FP8x23 { mag: 50331648, sign: true });
    data.append(FP8x23 { mag: 662700032, sign: true });
    data.append(FP8x23 { mag: 159383552, sign: false });
    data.append(FP8x23 { mag: 1015021568, sign: false });
    data.append(FP8x23 { mag: 142606336, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
