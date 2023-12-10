use core::traits::Into;
use orion::numbers::fixed_point::implementations::fp8x23wide::core::{ONE, FP8x23W, FixedTrait};
use orion::numbers::fixed_point::implementations::fp8x23wide::math::lut::erf_lut;

const ERF_COMPUTATIONAL_ACCURACY: u64 = 100;
const MAX_ERF_COMPUTATIONAL_ACCURACY: u64 = 10;
const ROUND_CHECK_NUMBER: u64 = 1;
// Values > MAX_ERF_NUMBER return 1
const MAX_ERF_NUMBER: u64 = 29360128;
// Values <= ERF_TRUNCATION_NUMBER -> two decimal places, and values > ERF_TRUNCATION_NUMBER -> one decimal place
const ERF_TRUNCATION_NUMBER: u64 = 16777216;

fn erf(x: FP8x23W) -> FP8x23W {
    // Lookup
    // 1. if x.mag < 3.5 { lookup table }
    // 2. else{ return 1}
    let mut erf_value: u64 = 0;

    if x.mag < MAX_ERF_NUMBER {
        erf_value = erf_lut(x.mag);
    } else {
        erf_value = ONE;
    }
    FP8x23W { mag: erf_value, sign: x.sign }
}
