use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7924942, sign: false });
    data.append(FP8x23 { mag: 13246850, sign: false });
    data.append(FP8x23 { mag: 2702202, sign: false });
    data.append(FP8x23 { mag: 10258009, sign: true });
    data.append(FP8x23 { mag: 1097270, sign: true });
    data.append(FP8x23 { mag: 3170256, sign: false });
    data.append(FP8x23 { mag: 11933079, sign: true });
    data.append(FP8x23 { mag: 7332484, sign: true });
    data.append(FP8x23 { mag: 10518663, sign: false });
    data.append(FP8x23 { mag: 2040333, sign: false });
    data.append(FP8x23 { mag: 11555977, sign: false });
    data.append(FP8x23 { mag: 192783, sign: true });
    data.append(FP8x23 { mag: 8410969, sign: false });
    data.append(FP8x23 { mag: 10253807, sign: false });
    data.append(FP8x23 { mag: 4451078, sign: false });
    data.append(FP8x23 { mag: 6128150, sign: false });
    data.append(FP8x23 { mag: 3066136, sign: true });
    data.append(FP8x23 { mag: 1979188, sign: false });
    data.append(FP8x23 { mag: 12213890, sign: true });
    data.append(FP8x23 { mag: 4054431, sign: false });
    data.append(FP8x23 { mag: 10368026, sign: false });
    data.append(FP8x23 { mag: 21004994, sign: true });
    data.append(FP8x23 { mag: 3484058, sign: true });
    data.append(FP8x23 { mag: 11422819, sign: false });
    data.append(FP8x23 { mag: 4590602, sign: false });
    data.append(FP8x23 { mag: 934665, sign: false });
    data.append(FP8x23 { mag: 13626899, sign: true });
    data.append(FP8x23 { mag: 2017121, sign: false });
    data.append(FP8x23 { mag: 3778004, sign: false });
    data.append(FP8x23 { mag: 12333339, sign: true });
    data.append(FP8x23 { mag: 6452644, sign: true });
    data.append(FP8x23 { mag: 23139458, sign: false });
    data.append(FP8x23 { mag: 9962093, sign: true });
    data.append(FP8x23 { mag: 28057456, sign: false });
    data.append(FP8x23 { mag: 4096432, sign: false });
    data.append(FP8x23 { mag: 1746466, sign: false });
    data.append(FP8x23 { mag: 3400610, sign: false });
    data.append(FP8x23 { mag: 1694851, sign: false });
    data.append(FP8x23 { mag: 14762852, sign: true });
    data.append(FP8x23 { mag: 11193272, sign: false });
    data.append(FP8x23 { mag: 3271310, sign: false });
    data.append(FP8x23 { mag: 8257251, sign: true });
    data.append(FP8x23 { mag: 5549365, sign: true });
    data.append(FP8x23 { mag: 1609902, sign: false });
    data.append(FP8x23 { mag: 9788646, sign: false });
    data.append(FP8x23 { mag: 4905918, sign: false });
    data.append(FP8x23 { mag: 7275039, sign: false });
    data.append(FP8x23 { mag: 15450536, sign: false });
    data.append(FP8x23 { mag: 7493020, sign: true });
    data.append(FP8x23 { mag: 5102934, sign: true });
    data.append(FP8x23 { mag: 4473274, sign: false });
    data.append(FP8x23 { mag: 12556968, sign: true });
    data.append(FP8x23 { mag: 10195289, sign: false });
    data.append(FP8x23 { mag: 15666986, sign: true });
    data.append(FP8x23 { mag: 5094279, sign: false });
    data.append(FP8x23 { mag: 17021828, sign: true });
    data.append(FP8x23 { mag: 6281825, sign: false });
    data.append(FP8x23 { mag: 3457968, sign: false });
    data.append(FP8x23 { mag: 4515889, sign: false });
    data.append(FP8x23 { mag: 3445940, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
