use orion_data_structures::SpanMathTrait;
use orion_numbers::FixedTrait;

/// Performs a linear least squares fitting to provided data points. This function calculates
/// the slope and intercept for the line of best fit using the formulae derived from the method
/// of least squares.
///
/// # Arguments
/// * `x` - A `Span<T>` containing the x-coordinates of the data points.
/// * `y` - A `Span<T>` containing the y-coordinates of the data points.
///
/// # Returns
/// A tuple `(T, T)` where the first element is the slope (`a`) and the second element is the
/// intercept (`b`) of the line of best fit.
///
/// # Panics
/// * The function panics if the lengths of `x` and `y` are not the same or if either is empty.
/// * It also panics if a division by zero occurs, which can happen if the points are perfectly
///   collinear along the x-axis resulting in zero variance in `x`.
///
/// # Examples
/// Basic usage:
///
/// ```
/// let x = array![0, 65536, 131072, 196608, 262144, 327680].span();
/// let y = array![0, 131072, 262144, 393216, 524288, 655360].span();
/// let (slope, intercept) = linear_fit(x, y);
/// // Expected output: (slope, intercept) where slope should closely match the true slope of the
/// data.
/// ```
///
/// With noisy data:
///
/// ```
/// let x = array![0, 65536, 131072, 196608, 262144, 327680].span();
/// let y = array![6554, 144179, 255590, 399770, 517734, 668467].span();
/// let (slope, intercept) = linear_fit(x, y);
/// // Expected output: (slope, intercept) that best fits the provided noisy data.
/// ```
pub fn linear_fit<
    T,
    S,
    +SpanMathTrait<T>,
    +TryInto<u32, T>,
    +FixedTrait<T, S>,
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

    let denominator = n * sum_xx - (sum_x * sum_x);
    if denominator == FixedTrait::ZERO() {
        panic!("division by zero exception")
    }

    let a = ((n * sum_xy) - sum_x * sum_y) / (denominator);
    let b = (sum_y - a * sum_x) / n;

    (a, b)
}

#[cfg(test)]
mod tests {
    use super::linear_fit;
    use orion_numbers::{F64, F64Impl, f64::helpers::assert_precise};

    #[test]
    fn linear_fit_line_test_f32x32() {
        let x = array![
            F64Impl::new(0),
            F64Impl::new(4294967296),
            F64Impl::new(8589934592),
            F64Impl::new(12884901888),
            F64Impl::new(17179869184),
            F64Impl::new(21474836480)
        ]
            .span();
        let y = array![
            F64Impl::new(0),
            F64Impl::new(8589934592),
            F64Impl::new(17179869184),
            F64Impl::new(25769803776),
            F64Impl::new(34359738368),
            F64Impl::new(42949672960)
        ]
            .span();

        let (slope_expected, intercept_expected) = (8589934592, 0);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        assert_precise(slope_actual, slope_expected, 'slopes should be equal', Option::None(()));
        assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );
    }

    #[test]
    fn linear_fit_line_with_noise_test_f32x32() {
        let x = array![
            F64Impl::new(0),
            F64Impl::new(4294967296),
            F64Impl::new(8589934592),
            F64Impl::new(12884901888),
            F64Impl::new(17179869184),
            F64Impl::new(21474836480)
        ]
            .span();
        let y = array![
            F64Impl::new(430014464),
            F64Impl::new(9448922319),
            F64Impl::new(16754909120),
            F64Impl::new(26198093840),
            F64Impl::new(33983924224),
            F64Impl::new(43786887168)
        ]
            .span();

        let (slope_expected, intercept_expected) = (8566644398, 350514194);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        assert_precise(slope_actual, slope_expected, 'slopes should be equal', Option::None(()));
        assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );
    }

    #[test]
    fn linear_fit_test_f32x32() {
        let x = array![
            F64Impl::new(0),
            F64Impl::new(4294967296),
            F64Impl::new(8589934592),
            F64Impl::new(12884901888),
            F64Impl::new(17179869184),
            F64Impl::new(21474836480)
        ]
            .span();
        let y = array![
            F64Impl::new(12458487808),
            F64Impl::new(12884901888),
            F64Impl::new(20111880192),
            F64Impl::new(21474836480),
            F64Impl::new(30031216640),
            F64Impl::new(47185920000)
        ]
            .span();

        let (slope_expected, intercept_expected) = (6479333376, 7850820812);
        let (slope_actual, intercept_actual) = linear_fit(x, y);

        assert_precise(slope_actual, slope_expected, 'slopes should be equal', Option::None(()));
        assert_precise(
            intercept_actual, intercept_expected, 'intercepts should be equal', Option::None(())
        );
    }
}
