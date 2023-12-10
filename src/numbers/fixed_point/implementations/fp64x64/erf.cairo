use core::traits::Into;
use orion::numbers::{FP64x64, FixedTrait};
use cubit::f128::ONE_u128 as ONE;

use orion::numbers::fixed_point::implementations::fp64x64::lut::erf_lut;

const ERF_COMPUTATIONAL_ACCURACY: u128 = 100_u128;
const ROUND_CHECK_NUMBER: u128 = 10_u128;
// Values > MAX_ERF_NUMBER return 1
const MAX_ERF_NUMBER: u128 = 64563604257983430656_u128;
// Values <= ERF_TRUNCATION_NUMBER -> two decimal places, and values > ERF_TRUNCATION_NUMBER -> one decimal place
const ERF_TRUNCATION_NUMBER: u128 = 36893488147419103232_u128;

fn erf(x: FP64x64) -> FP64x64 {
    // Lookup
    // 1. if x.mag < 3.5 { lookup table }
    // 2. else{ return 1}
    let mut erf_value: u128 = 0_u128;

    if x.mag <= MAX_ERF_NUMBER {
        erf_value = erf_lut(x.mag);
    } else {
        erf_value = ONE;
    }
    FP64x64 { mag: erf_value, sign: x.sign }
}
