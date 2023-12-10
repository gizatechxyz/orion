use orion::numbers::fixed_point::implementations::fp16x16wide::math::erf::erf;
use orion::numbers::fixed_point::implementations::fp16x16wide::core::{ONE, FP16x16W, FixedTrait};
use core::debug::PrintTrait;
#[test]
#[available_gas(1000000000)]
fn test_erf() {
    // 1.0
    let f1: FP16x16W = FP16x16W { mag: 65536, sign: false };
    // 0.134
    let f2: FP16x16W = FP16x16W { mag: 8832, sign: false };
    // 0.520
    let f3: FP16x16W = FP16x16W { mag: 34078, sign: false };
    // 2.0
    let f4: FP16x16W = FP16x16W { mag: 131072, sign: false };
    // 3.5
    let f5: FP16x16W = FP16x16W { mag: 229376, sign: false };
    // 5.164
    let f6: FP16x16W = FP16x16W { mag: 338428, sign: false };

    let f1_erf: FP16x16W = erf(f1);
    let f2_erf: FP16x16W = erf(f2);
    let f3_erf: FP16x16W = erf(f3);
    let f4_erf: FP16x16W = erf(f4);
    let f5_erf: FP16x16W = erf(f5);
    let f6_erf: FP16x16W = erf(f6);

    assert(f1_erf.mag == 55227, 'f1_erf it works!');
    assert(f2_erf.mag == 10285, 'f2_erf it works!');
    assert(f3_erf.mag == 35251, 'f3_erf it works!');
    assert(f4_erf.mag == 65229, 'f4_erf it works!');
    assert(f5_erf.mag == 65536, 'f5_erf it works!');
    assert(f6_erf.mag == 65536, 'f6_erf it works!');
}
