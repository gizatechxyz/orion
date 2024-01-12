use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 76012, sign: false });
    data.append(FP16x16 { mag: 98304, sign: false });
    data.append(FP16x16 { mag: 120595, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 102578, sign: false });
    data.append(FP16x16 { mag: 113054, sign: false });
    data.append(FP16x16 { mag: 135346, sign: false });
    data.append(FP16x16 { mag: 157637, sign: false });
    data.append(FP16x16 { mag: 168114, sign: false });
    data.append(FP16x16 { mag: 159565, sign: false });
    data.append(FP16x16 { mag: 170042, sign: false });
    data.append(FP16x16 { mag: 192333, sign: false });
    data.append(FP16x16 { mag: 214625, sign: false });
    data.append(FP16x16 { mag: 225101, sign: false });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 207084, sign: false });
    data.append(FP16x16 { mag: 229376, sign: false });
    data.append(FP16x16 { mag: 251667, sign: false });
    data.append(FP16x16 { mag: 262144, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
