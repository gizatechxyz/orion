use orion_numbers::f16x16::{core::{FixedTrait, f16x16, ONE}, lut};

const ERF_COMPUTATIONAL_ACCURACY: i32 = 100;
const ROUND_CHECK_NUMBER: i32 = 10;
// Values > MAX_ERF_NUMBER return 1
const MAX_ERF_NUMBER: i32 = 229376;
// Values <= ERF_TRUNCATION_NUMBER -> two decimal places, and values > ERF_TRUNCATION_NUMBER -> one
// decimal place
const ERF_TRUNCATION_NUMBER: i32 = 131072;

pub fn erf(x: f16x16) -> f16x16 {
    // Lookup
    // 1. if x.mag < 3.5 { lookup table }
    // 2. else{ return 1}
    let mut erf_value: i32 = 0;

    if x.abs() < MAX_ERF_NUMBER {
        erf_value = lut::erf_lut(x.abs());
    } else {
        erf_value = ONE;
    }

    FixedTrait::mul(erf_value, x.sign())
}


// Tests
//
// 
// --------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{erf, f16x16};

    #[test]
    #[available_gas(1000000000)]
    fn test_erf() {
        // 1.0
        let f1 = 65536;
        // 0.134
        let f2 = 8832;
        // 0.520
        let f3 = 34078;
        // 2.0
        let f4 = 131072;
        // 3.5
        let f5 = 229376;
        // 5.164
        let f6 = 338428;

        let f1_erf = erf(f1);
        let f2_erf = erf(f2);
        let f3_erf = erf(f3);
        let f4_erf = erf(f4);
        let f5_erf = erf(f5);
        let f6_erf = erf(f6);

        assert(f1_erf == 55227, 'f1_erf it works!');
        assert(f2_erf == 10285, 'f2_erf it works!');
        assert(f3_erf == 35251, 'f3_erf it works!');
        assert(f4_erf == 65229, 'f4_erf it works!');
        assert(f5_erf == 65536, 'f5_erf it works!');
        assert(f6_erf == 65536, 'f6_erf it works!');
    }
}
