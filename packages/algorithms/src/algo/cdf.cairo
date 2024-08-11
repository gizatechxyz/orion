use core::array::{SpanTrait, SpanIter};
use orion_numbers::FixedTrait;
use orion_data_structures::SpanMathTrait;

/// Computes the cumulative distribution function (CDF) for a given set of values using the
/// standard normal distribution formula. This implementation allows for optional location (`loc`)
/// and scale (`scale`) parameters, which default to 0.0 and 1.0 respectively if not provided.
///
/// # Arguments
/// * `x` - A `Span<T>` containing the data points for which the CDF is to be computed.
/// * `loc` - An optional `Span<T>` representing the location parameter (mean) for each data point.
///           If `Some(Span<T>)` is provided, it must either contain a single value or have the same
///           length as `x`. If `None` is provided, defaults to a Span of a single 0.0 value.
/// * `scale` - An optional `Span<T>` representing the scale parameter (standard deviation) for each
///             data point. If `Some(Span<T>)` is provided, it must either contain a single value
///             or have the same length as `x`. If `None` is provided, defaults to a Span of a
///             single 1.0 value.
///
/// # Returns
/// A `Span<T>` representing the CDF values corresponding to each entry in `x`.
///
/// # Panics
/// * The function panics if the lengths of `loc` or `scale` Spans are more than one and not equal
/// to
///   the length of `x`.
///
/// # Examples
/// Basic usage:
///
/// ```
/// let x = array![FixedTrait::new_unscaled(2), FixedTrait::new_unscaled(1),
/// FixedTrait::new_unscaled(0)].span();
/// let result = cdf(x, None, None);
/// // Expected output: CDF values for a standard normal distribution
/// ```
///
/// With location and scale parameters:
///
/// ```
/// let x = array![FixedTrait::new_unscaled(2), FixedTrait::new_unscaled(1),
/// FixedTrait::new_unscaled(0)].span();
/// let loc = array![FixedTrait::new_unscaled(1), FixedTrait::new_unscaled(1),
/// FixedTrait::new_unscaled(1)].span();
/// let scale = array![FixedTrait::new_unscaled(1), FixedTrait::new_unscaled(1),
/// FixedTrait::new_unscaled(1)].span();
/// let result = cdf(x, Some(loc), Some(scale));
/// // Expected output: Adjusted CDF values using specified loc and scale
/// ```
pub fn cdf<
    T,
    S,
    +FixedTrait<T, S>,
    +SpanMathTrait<T>,
    +Sub<T>,
    +Div<T>,
    +Mul<T>,
    +Drop<T>,
    +Add<T>,
    +Copy<T>
>(
    x: Span<T>, loc: Option<Span<T>>, scale: Option<Span<T>>
) -> Span<T> {
    // Default loc to 0.0 if not provided
    let mut loc = match loc {
        Option::Some(val) => val,
        Option::None => array![FixedTrait::ZERO()].span()
    };

    // Default scale to 1.0 if not provided
    let mut scale = match scale {
        Option::Some(val) => val,
        Option::None => array![FixedTrait::ONE()].span()
    };

    // single value or same length as x
    if loc.len() > 1 {
        assert!(loc.len() == x.len(), "`loc` must be a single value or same length as x");
    }

    // single value or same length as x
    if scale.len() > 1 {
        assert!(scale.len() == x.len(), "`scale` must be a single value or same length as x");
    }

    let mut res_data = array![];
    let mut x_copy = x;
    let loc_first_val = *loc.at(0);
    let scale_first_val = *scale.at(0);
    loop {
        match x_copy.pop_front() {
            Option::Some(x_val) => {
                let loc_val = if loc.len() > 1 {
                    *loc.pop_front().unwrap()
                } else {
                    loc_first_val
                };

                let scale_val = if scale.len() > 1 {
                    *scale.pop_front().unwrap()
                } else {
                    scale_first_val
                };

                // Calculate: 0.5 * (1.0 + erf((x_val - loc_val) / (scale_val * sqrt(2.0))))
                let calc = FixedTrait::HALF()
                    * (FixedTrait::ONE()
                        + ((*x_val - loc_val) / (scale_val * FixedTrait::TWO().sqrt())).erf());

                res_data.append(calc);
            },
            Option::None => { break; }
        }
    };

    res_data.span()
}

#[cfg(test)]
mod tests {
    use super::cdf;
    use orion_numbers::{F64, F64Impl, f64::helpers::{assert_relative_span, assert_relative}};
    #[test]
    fn test_cdf_loc_scale_are_none() {
        let x: Span<F64> = array![F64Impl::ONE(), F64Impl::HALF(), F64Impl::ZERO()].span();

        let res = cdf(x, Option::None, Option::None);
        let expected: Span<felt252> = array![3613548169, 2969808657, 2147483648].span();

        assert_relative_span(res, expected, 'res != expected', Option::Some(4294967));
    }

    #[test]
    fn test_cdf_loc_scale_are_some() {
        let x: Span<F64> = array![F64Impl::ONE(), F64Impl::HALF(), F64Impl::ZERO()].span();

        let loc: Span<F64> = array![F64Impl::HALF(), F64Impl::HALF(), F64Impl::HALF()].span();

        let scale: Span<F64> = array![F64Impl::HALF(), F64Impl::HALF(), F64Impl::HALF()].span();

        let res = cdf(x, Option::Some(loc), Option::Some(scale));
        let expected = array![3613548169, 2147483648, 681419126].span();

        assert_relative_span(res, expected, 'res != expected', Option::Some(4294967));
    }
}
