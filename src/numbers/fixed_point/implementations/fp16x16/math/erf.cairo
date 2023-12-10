use core::traits::Into;
use orion::numbers::fixed_point::implementations::fp16x16::core::{ONE, FP16x16, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::math::lut::erf_lut;

const ERF_COMPUTATIONAL_ACCURACY: u32 = 100;
const ROUND_CHECK_NUMBER: u32 = 10;
// Values > MAX_ERF_NUMBER return 1
const MAX_ERF_NUMBER: u32 = 229376;
// Values <= ERF_TRUNCATION_NUMBER -> two decimal places, and values > ERF_TRUNCATION_NUMBER -> one decimal place
const ERF_TRUNCATION_NUMBER: u32 = 131072;

fn erf(x: FP16x16) -> FP16x16 {
    // Lookup
    // 1. if x.mag < 3.5 { lookup table }
    // 2. else{ return 1}
    let mut erf_value: u32 = 0;

    if x.mag < MAX_ERF_NUMBER {
        erf_value = erf_lut(x.mag);
    } else {
        erf_value = ONE;
    }
    FP16x16 { mag: erf_value, sign: x.sign }
}
