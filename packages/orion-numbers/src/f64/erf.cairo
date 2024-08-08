use super::{F64, F64Impl, ONE};
use super::lut;

const ERF_COMPUTATIONAL_ACCURACY: i64 = 100;
const ROUND_CHECK_NUMBER: i64 = 10;
// Values > MAX_ERF_NUMBER return 1
const MAX_ERF_NUMBER: i64 = 15032385536;
// Values <= ERF_TRUNCATION_NUMBER -> two decimal places, and values > ERF_TRUNCATION_NUMBER -> one
// decimal place
const ERF_TRUNCATION_NUMBER: i64 = 8589934592;

pub(crate) fn erf(x: F64) -> F64 {
    // Lookup
    // 1. if x.mag < 3.5 { lookup table }
    // 2. else{ return 1}
    let mut erf_value: i64 = 0_i64;

    if x.d < MAX_ERF_NUMBER {
        erf_value = lut::erf_lut(x.d);
    } else {
        erf_value = ONE;
    }

    F64 { d: erf_value }
}

#[cfg(test)]
mod tests {
    use super::{F64, F64Impl, erf};

    #[test]
    #[available_gas(1000000000)]
    fn test_erf() {
        // 1.0
        let f1: F64 = F64 { d: 4294967296 };
        // 0.134
        let f2: F64 = F64 { d: 575525618 };
        // 0.520
        let f3: F64 = F64 { d: 2233382993 };
        // 2.0
        let f4: F64 = F64 { d: 8589934592 };
        // 3.5
        let f5: F64 = F64 { d: 15032385536 };
        // 5.164
        let f6: F64 = F64 { d: 22179211117 };

        let f1_erf: F64 = erf(f1);
        let f2_erf: F64 = erf(f2);
        let f3_erf: F64 = erf(f3);
        let f4_erf: F64 = erf(f4);
        let f5_erf: F64 = erf(f5);
        let f6_erf: F64 = erf(f6);
        assert(f1_erf.d == 3619372346, 'f1_erf does no work!');
        assert(f2_erf.d == 674082374, 'f2_erf does no work!');
        assert(f3_erf.d == 2310257026, 'f3_erf does no work!');
        assert(f4_erf.d == 4274876577, 'f4_erf does no work!');
        assert(f5_erf.d == 4294967296, 'f5_erf does no work!');
        assert(f6_erf.d == 4294967296, 'f6_erf does no work!');
    }
}
