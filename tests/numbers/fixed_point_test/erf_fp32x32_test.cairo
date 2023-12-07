use orion::numbers::fixed_point::implementations::fp32x32::erf::erf;
use orion::numbers::fixed_point::implementations::fp32x32::core::{
    ONE, FP32x32, FixedTrait
};
use debug::PrintTrait;
#[test]
#[available_gas(1000000000)]
fn test_erf() {
    // 1.0
    let f1: FP32x32 = FP32x32 { mag: 4294967296, sign: false };
    // 0.134
    let f2: FP32x32 = FP32x32 { mag: 575525618, sign: false };
    // 0.520
    let f3: FP32x32 = FP32x32 { mag: 2233382993, sign: false };
    // 2.0
    let f4: FP32x32 = FP32x32 { mag: 8589934592, sign: false };
    // 3.5
    let f5: FP32x32 = FP32x32 { mag: 15032385536, sign: false };
    // 5.164
    let f6: FP32x32 = FP32x32 { mag: 22179211117, sign: false };

    let f1_erf: FP32x32 = erf(f1);
    let f2_erf: FP32x32 = erf(f2);
    let f3_erf: FP32x32 = erf(f3);
    let f4_erf: FP32x32 = erf(f4);
    let f5_erf: FP32x32 = erf(f5);
    let f6_erf: FP32x32 = erf(f6);
    assert(f1_erf.mag == 3619372346, 'f1_erf it works!');
    assert(f2_erf.mag == 674082374, 'f2_erf it works!');
    assert(f3_erf.mag == 2310257026, 'f3_erf it works!');
    assert(f4_erf.mag == 4274876577, 'f4_erf it works!');
    assert(f5_erf.mag == 4294967296, 'f5_erf it works!');
    assert(f6_erf.mag == 4294967296, 'f6_erf it works!');

}