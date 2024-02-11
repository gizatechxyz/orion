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
    data.append(FP16x16 { mag: 1376256, sign: false });
    data.append(FP16x16 { mag: 3670016, sign: false });
    data.append(FP16x16 { mag: 851968, sign: false });
    data.append(FP16x16 { mag: 1048576, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 4128768, sign: false });
    data.append(FP16x16 { mag: 2293760, sign: false });
    data.append(FP16x16 { mag: 4390912, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 917504, sign: false });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 1441792, sign: false });
    data.append(FP16x16 { mag: 4980736, sign: false });
    data.append(FP16x16 { mag: 4980736, sign: false });
    data.append(FP16x16 { mag: 1376256, sign: false });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 5767168, sign: false });
    data.append(FP16x16 { mag: 2949120, sign: false });
    data.append(FP16x16 { mag: 4128768, sign: false });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 2162688, sign: false });
    data.append(FP16x16 { mag: 1179648, sign: false });
    data.append(FP16x16 { mag: 3538944, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
