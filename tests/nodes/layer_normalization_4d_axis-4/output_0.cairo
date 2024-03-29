use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 81938, sign: false });
    data.append(FP16x16 { mag: 139115, sign: true });
    data.append(FP16x16 { mag: 99625, sign: false });
    data.append(FP16x16 { mag: 42553, sign: false });
    data.append(FP16x16 { mag: 63807, sign: true });
    data.append(FP16x16 { mag: 29251, sign: false });
    data.append(FP16x16 { mag: 74903, sign: true });
    data.append(FP16x16 { mag: 9937, sign: true });
    data.append(FP16x16 { mag: 65116, sign: true });
    data.append(FP16x16 { mag: 48943, sign: false });
    data.append(FP16x16 { mag: 37524, sign: false });
    data.append(FP16x16 { mag: 77082, sign: false });
    data.append(FP16x16 { mag: 57309, sign: false });
    data.append(FP16x16 { mag: 3013, sign: true });
    data.append(FP16x16 { mag: 22472, sign: false });
    data.append(FP16x16 { mag: 68481, sign: false });
    data.append(FP16x16 { mag: 47521, sign: true });
    data.append(FP16x16 { mag: 29144, sign: false });
    data.append(FP16x16 { mag: 11016, sign: false });
    data.append(FP16x16 { mag: 63886, sign: true });
    data.append(FP16x16 { mag: 88094, sign: false });
    data.append(FP16x16 { mag: 112894, sign: true });
    data.append(FP16x16 { mag: 7732, sign: false });
    data.append(FP16x16 { mag: 45680, sign: true });
    data.append(FP16x16 { mag: 52166, sign: false });
    data.append(FP16x16 { mag: 135141, sign: true });
    data.append(FP16x16 { mag: 18324, sign: true });
    data.append(FP16x16 { mag: 201528, sign: false });
    data.append(FP16x16 { mag: 125050, sign: false });
    data.append(FP16x16 { mag: 50933, sign: true });
    data.append(FP16x16 { mag: 64223, sign: true });
    data.append(FP16x16 { mag: 52706, sign: false });
    data.append(FP16x16 { mag: 106918, sign: false });
    data.append(FP16x16 { mag: 134854, sign: false });
    data.append(FP16x16 { mag: 6678, sign: false });
    data.append(FP16x16 { mag: 100030, sign: true });
    data.append(FP16x16 { mag: 8001, sign: false });
    data.append(FP16x16 { mag: 75425, sign: true });
    data.append(FP16x16 { mag: 3641, sign: false });
    data.append(FP16x16 { mag: 20349, sign: true });
    data.append(FP16x16 { mag: 33423, sign: true });
    data.append(FP16x16 { mag: 135434, sign: false });
    data.append(FP16x16 { mag: 268826, sign: true });
    data.append(FP16x16 { mag: 9384, sign: false });
    data.append(FP16x16 { mag: 42989, sign: true });
    data.append(FP16x16 { mag: 40596, sign: false });
    data.append(FP16x16 { mag: 27643, sign: true });
    data.append(FP16x16 { mag: 14881, sign: false });
    data.append(FP16x16 { mag: 33184, sign: false });
    data.append(FP16x16 { mag: 245527, sign: false });
    data.append(FP16x16 { mag: 12844, sign: false });
    data.append(FP16x16 { mag: 36484, sign: false });
    data.append(FP16x16 { mag: 43540, sign: true });
    data.append(FP16x16 { mag: 98782, sign: false });
    data.append(FP16x16 { mag: 91559, sign: true });
    data.append(FP16x16 { mag: 175239, sign: true });
    data.append(FP16x16 { mag: 36924, sign: true });
    data.append(FP16x16 { mag: 69880, sign: true });
    data.append(FP16x16 { mag: 218008, sign: false });
    data.append(FP16x16 { mag: 89765, sign: true });
    data.append(FP16x16 { mag: 37583, sign: false });
    data.append(FP16x16 { mag: 40273, sign: false });
    data.append(FP16x16 { mag: 47428, sign: true });
    data.append(FP16x16 { mag: 33322, sign: true });
    data.append(FP16x16 { mag: 53002, sign: true });
    data.append(FP16x16 { mag: 51571, sign: false });
    data.append(FP16x16 { mag: 43023, sign: true });
    data.append(FP16x16 { mag: 180611, sign: false });
    data.append(FP16x16 { mag: 27436, sign: false });
    data.append(FP16x16 { mag: 82088, sign: true });
    data.append(FP16x16 { mag: 49075, sign: true });
    data.append(FP16x16 { mag: 93116, sign: true });
    data.append(FP16x16 { mag: 24713, sign: true });
    data.append(FP16x16 { mag: 57986, sign: true });
    data.append(FP16x16 { mag: 11641, sign: false });
    data.append(FP16x16 { mag: 87906, sign: true });
    data.append(FP16x16 { mag: 73854, sign: false });
    data.append(FP16x16 { mag: 22022, sign: true });
    data.append(FP16x16 { mag: 31528, sign: true });
    data.append(FP16x16 { mag: 151422, sign: true });
    data.append(FP16x16 { mag: 270761, sign: true });
    data.append(FP16x16 { mag: 37881, sign: false });
    data.append(FP16x16 { mag: 81795, sign: true });
    data.append(FP16x16 { mag: 65199, sign: false });
    data.append(FP16x16 { mag: 89952, sign: true });
    data.append(FP16x16 { mag: 36152, sign: false });
    data.append(FP16x16 { mag: 20482, sign: true });
    data.append(FP16x16 { mag: 296268, sign: true });
    data.append(FP16x16 { mag: 79775, sign: true });
    data.append(FP16x16 { mag: 66513, sign: false });
    data.append(FP16x16 { mag: 107842, sign: false });
    data.append(FP16x16 { mag: 37504, sign: false });
    data.append(FP16x16 { mag: 170624, sign: false });
    data.append(FP16x16 { mag: 52118, sign: false });
    data.append(FP16x16 { mag: 43525, sign: false });
    data.append(FP16x16 { mag: 138068, sign: false });
    data.append(FP16x16 { mag: 92822, sign: true });
    data.append(FP16x16 { mag: 155459, sign: true });
    data.append(FP16x16 { mag: 101038, sign: false });
    data.append(FP16x16 { mag: 45944, sign: false });
    data.append(FP16x16 { mag: 14639, sign: true });
    data.append(FP16x16 { mag: 30883, sign: true });
    data.append(FP16x16 { mag: 262024, sign: false });
    data.append(FP16x16 { mag: 30507, sign: true });
    data.append(FP16x16 { mag: 43715, sign: true });
    data.append(FP16x16 { mag: 84662, sign: true });
    data.append(FP16x16 { mag: 63901, sign: false });
    data.append(FP16x16 { mag: 3913, sign: false });
    data.append(FP16x16 { mag: 40400, sign: true });
    data.append(FP16x16 { mag: 36018, sign: true });
    data.append(FP16x16 { mag: 49780, sign: true });
    data.append(FP16x16 { mag: 146468, sign: false });
    data.append(FP16x16 { mag: 32554, sign: true });
    data.append(FP16x16 { mag: 39291, sign: true });
    data.append(FP16x16 { mag: 93325, sign: false });
    data.append(FP16x16 { mag: 216, sign: false });
    data.append(FP16x16 { mag: 67978, sign: true });
    data.append(FP16x16 { mag: 97710, sign: false });
    data.append(FP16x16 { mag: 19520, sign: false });
    data.append(FP16x16 { mag: 6998, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
