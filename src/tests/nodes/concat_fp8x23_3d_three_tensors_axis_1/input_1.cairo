use array::ArrayTrait;

use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp8x23::Tensor_fp8x23;use orion::numbers::FP8x23;


fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 226492416, sign: false });
    data.append(FP8x23 { mag: 234881024, sign: false });
    data.append(FP8x23 { mag: 243269632, sign: false });
    data.append(FP8x23 { mag: 251658240, sign: false });
    data.append(FP8x23 { mag: 260046848, sign: false });
    data.append(FP8x23 { mag: 268435456, sign: false });
    data.append(FP8x23 { mag: 276824064, sign: false });
    data.append(FP8x23 { mag: 285212672, sign: false });
    data.append(FP8x23 { mag: 293601280, sign: false });
    data.append(FP8x23 { mag: 301989888, sign: false });
    data.append(FP8x23 { mag: 310378496, sign: false });
    data.append(FP8x23 { mag: 318767104, sign: false });
    data.append(FP8x23 { mag: 327155712, sign: false });
    data.append(FP8x23 { mag: 335544320, sign: false });
    data.append(FP8x23 { mag: 343932928, sign: false });
    data.append(FP8x23 { mag: 352321536, sign: false });
    data.append(FP8x23 { mag: 360710144, sign: false });
    data.append(FP8x23 { mag: 369098752, sign: false });
    data.append(FP8x23 { mag: 377487360, sign: false });
    data.append(FP8x23 { mag: 385875968, sign: false });
    data.append(FP8x23 { mag: 394264576, sign: false });
    data.append(FP8x23 { mag: 402653184, sign: false });
    data.append(FP8x23 { mag: 411041792, sign: false });
    data.append(FP8x23 { mag: 419430400, sign: false });
    data.append(FP8x23 { mag: 427819008, sign: false });
    data.append(FP8x23 { mag: 436207616, sign: false });
    data.append(FP8x23 { mag: 444596224, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}