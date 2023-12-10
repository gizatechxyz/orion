use orion::numbers::fixed_point::implementations::fp64x64::erf::erf;
use orion::numbers::fixed_point::implementations::fp64x64::core::{
    ONE, FP64x64, FixedTrait
};
use core::debug::PrintTrait;
#[test]
#[available_gas(1000000000)]
fn test_erf() {
    // 1.0
    let f1: FP64x64 = FP64x64 { mag: 18446744073709551616_u128, sign: false };
    // 0.134
    let f2: FP64x64 = FP64x64 { mag: 2471863705877080064_u128, sign: false };
    // 0.520
    let f3: FP64x64 = FP64x64 { mag: 9592306918328967168_u128, sign: false };
    // 2.0
    let f4: FP64x64 = FP64x64 { mag: 36893488147419103232_u128, sign: false };
    // 3.5
    let f5: FP64x64 = FP64x64 { mag: 64563604257983430656_u128, sign: false };
    // 5.164
    let f6: FP64x64 = FP64x64 { mag: 95258986396636119040_u128, sign: false };

    let f1_erf: FP64x64 = erf(f1);
    let f2_erf: FP64x64 = erf(f2);
    let f3_erf: FP64x64 = erf(f3);
    let f4_erf: FP64x64 = erf(f4);
    let f5_erf: FP64x64 = erf(f5);
    let f6_erf: FP64x64 = erf(f6);
    assert(f1_erf.mag == 15545085858255493120_u128, 'f1_erf it works!');
    assert(f2_erf.mag == 2895161752038532608_u128, 'f2_erf it works!');
    assert(f3_erf.mag == 9922478374042292224_u128, 'f3_erf it works!');
    assert(f4_erf.mag == 18360455093669533696_u128, 'f4_erf it works!');
    assert(f5_erf.mag == 18446744073709551616_u128, 'f5_erf it works!');
    assert(f6_erf.mag == 18446744073709551616_u128, 'f6_erf it works!');

}