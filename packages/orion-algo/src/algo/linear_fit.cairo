use orion_numbers::{f16x16::core::{f16x16}, FixedTrait};
use orion_algo::span_math::SpanMathTrait;
use orion_numbers::core_trait::{I32Div, I64Div};


pub fn linear_fit<
    T,
    +SpanMathTrait<T>,
    +TryInto<u32, T>,
    +FixedTrait<T>,
    +Mul<T>,
    +Sub<T>,
    +PartialEq<T>,
    +Div<T>,
    +Drop<T>,
    +Copy<T>
>(
    x: Span<T>, y: Span<T>
) -> (T, T) {
    if x.len() != y.len() || x.len() == 0 {
        panic!("x and y should be of the same lenght")
    }

    let n: T = x.len().try_into().unwrap();
    let sum_x = x.sum();
    let sum_y = y.sum();
    let sum_xx = x.dot(x);
    let sum_xy = x.dot(y);

    let denominator = n * sum_xx - (sum_x.mul(sum_x));
    if denominator == FixedTrait::ZERO() {
        panic!("division by zero exception")
    }

    let a = ((n * sum_xy) - sum_x.mul(sum_y)).div(denominator);
    let b = (sum_y - a.mul(sum_x)) / n;

    (a, b)
}

#[cfg(test)]
mod tests {
    use super::linear_fit;
    use orion_numbers::f16x16;
    use orion_numbers::f32x32;
    use orion_numbers::core_trait::{I32Div, I64Div};

    #[test]
    fn linear_fit_line_test() {
        let x = array![0, 65536, 131072, 196608, 262144, 327680].span();
        let y = array![0, 131072, 262144, 393216, 524288, 655360].span();

        let (slope_expected, intercept_expected) = (131072, 0);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        f16x16::helpers::assert_precise(
            slope_actual, slope_expected, 'slopes should be equal', Option::None(())
        );
        f16x16::helpers::assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );
    }

    #[test]
    fn linear_fit_line_with_noise_test() {
        let x = array![0, 65536, 131072, 196608, 262144, 327680].span();
        let y = array![6554, 144179, 255590, 399770, 517734, 668467].span();

        let (slope_expected, intercept_expected) = (130698, 5305);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        f16x16::helpers::assert_precise(
            slope_actual, slope_expected, 'slopes should be equal', Option::None(())
        );
        f16x16::helpers::assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );
    }

    #[test]
    fn linear_fit_test() {
        let x = array![0, 65536, 131072, 196608, 262144, 327680].span();
        let y = array![190054, 196608, 308019, 327680, 458752, 720896].span();

        let (slope_expected, intercept_expected) = (98866, 119837);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        f16x16::helpers::assert_precise(
            slope_actual, slope_expected, 'slopes should be equal', Option::None(())
        );
        f16x16::helpers::assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );
    }


    #[test]
    fn linear_fit_line_test_f32x32() {
        let x = array![0, 4294967296, 8589934592, 12884901888, 17179869184, 21474836480].span();
        let y = array![0, 8589934592, 17179869184, 25769803776, 34359738368, 42949672960].span();

        let (slope_expected, intercept_expected) = (8589934592, 0);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        f32x32::helpers::assert_precise(
            slope_actual, slope_expected, 'slopes should be equal', Option::None(())
        );
        f32x32::helpers::assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );
    }

    #[test]
    fn linear_fit_line_with_noise_test_f32x32() {
        let x = array![0, 4294967296, 8589934592, 12884901888, 17179869184, 21474836480].span();
        let y = array![430014464, 9448922319, 16754909120, 26198093840, 33983924224, 43786887168]
            .span();

        let (slope_expected, intercept_expected) = (8566644398, 350514194);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        f32x32::helpers::assert_precise(
            slope_actual, slope_expected, 'slopes should be equal', Option::None(())
        );
        f32x32::helpers::assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );
    }

    #[test]
    fn linear_fit_test_f32x32() {
        let x = array![0, 4294967296, 8589934592, 12884901888, 17179869184, 21474836480].span();
        let y = array![12458487808, 12884901888, 20111880192, 21474836480, 30031216640, 47185920000]
            .span();

        let (slope_expected, intercept_expected) = (6479333376, 7850820812);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        f32x32::helpers::assert_precise(
            slope_actual, slope_expected, 'slopes should be equal', Option::None(())
        );
        f32x32::helpers::assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );
    }
}
