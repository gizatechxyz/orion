use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7376595, sign: false });
    data.append(FP8x23 { mag: 3368567, sign: true });
    data.append(FP8x23 { mag: 7364417, sign: true });
    data.append(FP8x23 { mag: 14828661, sign: true });
    data.append(FP8x23 { mag: 12112126, sign: false });
    data.append(FP8x23 { mag: 11566799, sign: false });
    data.append(FP8x23 { mag: 2580433, sign: true });
    data.append(FP8x23 { mag: 3725608, sign: false });
    data.append(FP8x23 { mag: 9229898, sign: false });
    data.append(FP8x23 { mag: 17252748, sign: false });
    data.append(FP8x23 { mag: 17297774, sign: true });
    data.append(FP8x23 { mag: 12381053, sign: true });
    data.append(FP8x23 { mag: 18783952, sign: false });
    data.append(FP8x23 { mag: 19849344, sign: false });
    data.append(FP8x23 { mag: 4463606, sign: false });
    data.append(FP8x23 { mag: 6833714, sign: false });
    data.append(FP8x23 { mag: 5433502, sign: true });
    data.append(FP8x23 { mag: 12918222, sign: false });
    data.append(FP8x23 { mag: 14840246, sign: true });
    data.append(FP8x23 { mag: 8510834, sign: true });
    data.append(FP8x23 { mag: 10133696, sign: false });
    data.append(FP8x23 { mag: 16163513, sign: false });
    data.append(FP8x23 { mag: 4889398, sign: false });
    data.append(FP8x23 { mag: 1966734, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
