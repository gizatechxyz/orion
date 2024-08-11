use core::ops::AddAssign;

use orion_numbers::{FixedTrait};

use orion_algo::span_math::SpanMathTrait;


pub impl FixedSpanMath<
    T,
    S,
    +FixedTrait<T, S>,
    +Into<u32, S>,
    +Drop<T>,
    +Copy<T>,
    +Add<T>,
    +Mul<T>,
    +AddAssign<T, T>,
    +PartialOrd<T>
> of SpanMathTrait<T> {
    fn arange(n: u32) -> Span<T> {
        let mut i = 0;
        let mut arr = array![];
        while i < n {
            arr.append(FixedTrait::new_unscaled(i.into()));
            i += 1;
        };

        arr.span()
    }

    fn dot(self: Span<T>, other: Span<T>) -> T {
        let mut i = 0;
        let mut acc = FixedTrait::ZERO();
        while i != self.len() {
            acc += *self.at(i) * *other.at(i);
            i += 1;
        };

        acc
    }

    fn max(self: Span<T>) -> T {
        assert(self.len() > 0, 'span cannot be empty');

        let mut max = FixedTrait::MIN();
        let mut self = self;
        loop {
            match self.pop_front() {
                Option::Some(item) => { if *item > max {
                    max = *item;
                } },
                Option::None => { break max; },
            }
        }
    }

    fn min(self: Span<T>) -> T {
        assert(self.len() > 0, 'span cannot be empty');

        let mut min = FixedTrait::MAX();
        let mut self = self;
        loop {
            match self.pop_front() {
                Option::Some(item) => { if *item < min {
                    min = *item;
                } },
                Option::None => { break min; },
            }
        }
    }

    fn prod(self: Span<T>) -> T {
        let mut prod = FixedTrait::ONE();
        let mut self = self;
        loop {
            match self.pop_front() {
                Option::Some(v) => { prod = prod * *v; },
                Option::None => { break prod; }
            };
        }
    }

    fn sum(self: Span<T>) -> T {
        let mut prod = FixedTrait::ZERO();
        let mut self = self;
        loop {
            match self.pop_front() {
                Option::Some(v) => { prod = prod + *v; },
                Option::None => { break prod; }
            };
        }
    }
}


#[cfg(test)]
mod tests {
    use super::{FixedSpanMath, FixedTrait};
    use orion_numbers::{F64, F64Impl, f64::helpers::assert_precise};


    #[test]
    fn test_arange() {
        let n = 6;
        let res = FixedSpanMath::arange(n);

        let x = array![0, 4294967296, 8589934592, 12884901888, 17179869184, 21474836480].span();

        assert_precise(*res.at(0), *x.at(0), 'should be equal', Option::None);
        assert_precise(*res.at(1), *x.at(1), 'should be equal', Option::None);
        assert_precise(*res.at(2), *x.at(2), 'should be equal', Option::None);
        assert_precise(*res.at(3), *x.at(3), 'should be equal', Option::None);
        assert_precise(*res.at(4), *x.at(4), 'should be equal', Option::None);
        assert_precise(*res.at(5), *x.at(5), 'should be equal', Option::None);
    }

    #[test]
    fn test_dot() {
        let x = array![
            F64Impl::new(0),
            F64Impl::new(4294967296),
            F64Impl::new(8589934592),
            F64Impl::new(12884901888),
            F64Impl::new(17179869184),
            F64Impl::new(21474836480)
        ]
            .span(); // 0, 1, 2, 3, 4, 5
        let y = array![
            F64Impl::new(0),
            F64Impl::new(8589934592),
            F64Impl::new(17179869184),
            F64Impl::new(25769803776),
            F64Impl::new(34359738368),
            F64Impl::new(42949672960)
        ]
            .span(); // 0, 2, 4, 6, 8, 10
        let result = FixedSpanMath::dot(x, y);

        assert_precise(
            result, (F64Impl::new(472446402560)).into(), 'should be equal', Option::None
        );
    }

    #[test]
    fn test_max() {
        let x = array![
            F64Impl::new(0),
            F64Impl::new(4294967296),
            F64Impl::new(8589934592),
            F64Impl::new(12884901888),
            F64Impl::new(17179869184),
            F64Impl::new(21474836480)
        ]
            .span(); // 0, 1, 2, 3, 4, 5

        let result = FixedSpanMath::max(x);

        assert_precise(result, 21474836480, 'should be equal', Option::None);
    }

    #[test]
    fn test_min() {
        let x = array![
            F64Impl::new(0),
            F64Impl::new(4294967296),
            F64Impl::new(8589934592),
            F64Impl::new(12884901888),
            F64Impl::new(17179869184),
            F64Impl::new(21474836480)
        ]
            .span(); // 0, 1, 2, 3, 4, 5

        let result = FixedSpanMath::min(x);

        assert_precise(result, 0, 'should be equal', Option::None);
    }

    #[test]
    fn test_prod() {
        let x = array![
            F64Impl::new(0),
            F64Impl::new(4294967296),
            F64Impl::new(8589934592),
            F64Impl::new(12884901888),
            F64Impl::new(17179869184),
            F64Impl::new(21474836480)
        ]
            .span(); // 0, 1, 2, 3, 4, 5

        let result = FixedSpanMath::prod(x);

        assert_precise(result, 0, 'should be equal', Option::None);
    }
    #[test]
    fn test_sum() {
        let x = array![
            F64Impl::new(0),
            F64Impl::new(4294967296),
            F64Impl::new(8589934592),
            F64Impl::new(12884901888),
            F64Impl::new(17179869184),
            F64Impl::new(21474836480)
        ]
            .span(); // 0, 1, 2, 3, 4, 5

        let result = FixedSpanMath::sum(x);

        assert_precise(result, 64424509440, 'should be equal', Option::None);
    }
}
