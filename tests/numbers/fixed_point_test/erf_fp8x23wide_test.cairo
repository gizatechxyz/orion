use orion::numbers::fixed_point::implementations::fp8x23wide::math::erf::erf;
use orion::numbers::fixed_point::implementations::fp8x23wide::core::{
    ONE, FP8x23W, FixedTrait
};
use debug::PrintTrait;
#[test]
#[available_gas(1000000000)]
fn test_erf() {
    // 1.0
    let f1: FP8x23W = FP8x23W { mag: 8388608, sign: false };
    // 0.134
    let f2: FP8x23W = FP8x23W { mag: 1124073, sign: false };
    // 0.520
    let f3: FP8x23W = FP8x23W { mag: 4362076, sign: false };
    // 2.0
    let f4: FP8x23W = FP8x23W { mag: 16777216, sign: false };
    // 3.5
    let f5: FP8x23W = FP8x23W { mag: 29360128, sign: false };
    // 5.164
    let f6: FP8x23W = FP8x23W { mag: 43318772, sign: false };

    let f1_erf: FP8x23W = erf(f1);
    let f2_erf: FP8x23W = erf(f2);
    let f3_erf: FP8x23W = erf(f3);
    let f4_erf: FP8x23W = erf(f4);
    let f5_erf: FP8x23W = erf(f5);
    let f6_erf: FP8x23W = erf(f6);

    assert(f1_erf.mag == 7069086, 'f1_erf it works!');
    assert(f2_erf.mag == 1316567, 'f2_erf it works!');
    assert(f3_erf.mag == 4512220, 'f3_erf it works!');
    assert(f4_erf.mag == 8349368, 'f4_erf it works!');
    assert(f5_erf.mag == 8388608, 'f5_erf it works!');
    assert(f6_erf.mag == 8388608, 'f6_erf it works!');

}