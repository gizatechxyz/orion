use orion_numbers::{core_trait::{I64Rem, I64Div}, FixedTrait};
use orion_numbers::f32x32::core::{f32x32, ONE};

use orion_algo::span_math::SpanMathTrait;


pub impl F32x32SpanMath of SpanMathTrait<f32x32> {
    fn arange(n: u32) -> Span<f32x32> {
        arange(n)
    }

    fn dot(self: Span<f32x32>, other: Span<f32x32>) -> f32x32 {
        dot(self, other)
    }

    fn max(self: Span<f32x32>) -> f32x32 {
        max(self)
    }

    fn min(self: Span<f32x32>) -> f32x32 {
        min(self)
    }

    fn prod(self: Span<f32x32>) -> f32x32 {
        prod(self)
    }

    fn sum(self: Span<f32x32>) -> f32x32 {
        sum(self)
    }
}

fn arange(n: u32) -> Span<f32x32> {
    let mut i = 0;
    let mut arr = array![];
    while i < n {
        arr.append(i.try_into().unwrap() * ONE);
        i += 1;
    };

    arr.span()
}

fn dot(a: Span<f32x32>, b: Span<f32x32>) -> f32x32 {
    let mut i = 0;
    let mut acc = 0;
    while i != a.len() {
        acc += FixedTrait::mul(*a.at(i), *b.at(i));
        i += 1;
    };

    acc
}

fn max(mut a: Span<f32x32>) -> f32x32 {
    assert(a.len() > 0, 'span cannot be empty');

    let mut max = FixedTrait::MIN();

    loop {
        match a.pop_front() {
            Option::Some(item) => { if *item > max {
                max = *item;
            } },
            Option::None => { break max; },
        }
    }
}

fn min(mut a: Span<f32x32>) -> f32x32 {
    assert(a.len() > 0, 'span cannot be empty');

    let mut min = FixedTrait::MAX();

    loop {
        match a.pop_front() {
            Option::Some(item) => { if *item < min {
                min = *item;
            } },
            Option::None => { break min; },
        }
    }
}

fn prod(mut a: Span<f32x32>) -> f32x32 {
    let mut prod = 1;
    loop {
        match a.pop_front() {
            Option::Some(v) => { prod = prod.mul(*v); },
            Option::None => { break prod; }
        };
    }
}

fn sum(mut a: Span<f32x32>) -> f32x32 {
    let mut prod = 1;
    loop {
        match a.pop_front() {
            Option::Some(v) => { prod = prod + *v; },
            Option::None => { break prod; }
        };
    }
}


pub fn linear_fit(x: Span<f32x32>, y: Span<f32x32>) -> (f32x32, f32x32) {
    if x.len() != y.len() || x.len() == 0 {
        panic!("x and y should be of the same lenght")
    }

    let n: f32x32 = x.len().try_into().unwrap();
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
