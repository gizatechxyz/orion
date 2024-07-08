use orion_numbers::{f16x16::core::{f16x16}, FixedTrait};
use orion_algo::span_math::SpanMathTrait;
use orion_numbers::core_trait::I32Div;

pub fn linear_fit(x: Span<f16x16>, y: Span<f16x16>) -> (f16x16, f16x16) {
    if x.len() != y.len() || x.len() == 0 {
        panic!("x and y should be of the same lenght")
    }

    let n: f16x16 = x.len().try_into().unwrap();
    let sum_x = x.sum();
    let sum_y = y.sum();
    let sum_xx = x.dot(x);
    let sum_xy = x.dot(y);

    let denominator = n * sum_xx - (sum_x.mul(sum_x));
    if denominator == 0 {
        panic!("division by zero exception")
    }

    let a = ((n * sum_xy) - sum_x.mul(sum_y)).div(denominator);
    let b = (sum_y - a.mul(sum_x)) / n;

    (a, b)
}

#[cfg(test)]
mod tests {
    use super::linear_fit;
    use orion_numbers::f16x16::helpers::{assert_precise, assert_relative};

    #[test]
    fn linear_fit_line_test() {
        let x = array![0, 65536, 131072, 196608, 262144, 327680].span();
        let y = array![0, 131072, 262144, 393216, 524288, 655360].span();

        let (slope_expected, intercept_expected) = (131072, 0);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        assert_precise(
            slope_actual, slope_expected, 'slopes should be equal', Option::None(())
        );
        assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );
    }

    #[test]
    fn linear_fit_line_with_noise_test() {
        let x = array![0, 65536, 131072, 196608, 262144, 327680].span();
        let y = array![6554, 144179, 255590, 399770, 517734, 668467].span();

        let (slope_expected, intercept_expected) = (130698, 5305);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        assert_precise(
            slope_actual, slope_expected, 'slopes should be equal', Option::None(())
        );
        assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );

    }

    #[test]
    fn linear_fit_test() {
        let x = array![0, 65536, 131072, 196608, 262144, 327680].span();
        let y = array![190054, 196608, 308019, 327680, 458752, 720896].span();

        let (slope_expected, intercept_expected) = (98866, 119837);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        assert_precise(
            slope_actual, slope_expected, 'slopes should be equal', Option::None(())
        );
        assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );
    }

}
