use cubit::f64::ONE;

use orion::numbers::{FP32x32, FixedTrait};
use orion::numbers::fixed_point::implementations::fp32x32::lut::erf_lut;

const ERF_COMPUTATIONAL_ACCURACY: u64 = 100;
const ROUND_CHECK_NUMBER: u64 = 10;
// Values > MAX_ERF_NUMBER return 1
const MAX_ERF_NUMBER: u64 = 15032385536;
// Values <= ERF_TRUNCATION_NUMBER -> two decimal places, and values > ERF_TRUNCATION_NUMBER -> one decimal place
const ERF_TRUNCATION_NUMBER: u64 = 8589934592;

fn erf(x: FP32x32) -> FP32x32 {
    // Lookup
    // 1. if x.mag < 3.5 { lookup table }
    // 2. else{ return 1}
    let mut erf_value: u64 = 0_u64;

    if x.mag < MAX_ERF_NUMBER {
        erf_value = erf_lut(x.mag);
    } else {
        erf_value = ONE;
    }

    FP32x32 { mag: erf_value, sign: x.sign }
}
