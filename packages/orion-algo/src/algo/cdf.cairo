use core::clone::Clone;
use core::array::{SpanTrait, SpanIter};
use orion_algo::span_math::SpanMathTrait;
use orion_numbers::FixedTrait;

pub fn cdf<
    T, +FixedTrait<T>, +SpanMathTrait<T>, +Sub<T>, +Div<T>, +Mul<T>, +Drop<T>, +Add<T>, +Copy<T>
>(
    x: Span<T>, loc: Option<Span<T>>, scale: Option<Span<T>>
) //-> Span<T>
{
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
        assert_eq!(loc.len(), x.len(), "`loc` must be a single value or same length as x");
    }

    // single value or same length as x
    if scale.len() > 1 {
        assert_eq!(scale.len(), x.len(), "`scale` must be a single value or same length as x");
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

                //  Calculate: 0.5 * (1.0 + erf((x_val - loc_val) / (scale_val * (2.0f64).sqrt())))
                let calc = FixedTrait::HALF()
                    * (FixedTrait::ONE()
                        + ((*x_val - loc_val)
                            / (scale_val * (FixedTrait::ONE() + FixedTrait::ONE()).sqrt()))
                            .erf());

                res_data.append(calc);
            },
            Option::None => { break; }
        }
    };
}
