use core::array::ArrayTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

#[derive(Copy, Drop)]
enum MODE {
    STANDARD,
    NESTEROV
}

/// Cf: TensorTrait::momentum docstring
fn momentum<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +Sub<T>,
    +PartialOrd<T>,
>(
    r: T, t: T, inputs: @Tensor<T>, alpha: T, beta: T, mode: MODE, norm_coefficient: T,
) -> (Tensor<T>, Tensor<T>) {
    if (*inputs).data.len() == 3 {
        let (x, v) = run_momentum(
            r,
            t,
            *(*inputs).data.at(0),
            *(*inputs).data.at(1),
            *(*inputs).data.at(2),
            alpha,
            beta,
            mode,
            norm_coefficient
        );
        return (
            TensorTrait::new(array![1].span(), array![x].span()),
            TensorTrait::new(array![1].span(), array![v].span())
        );
    }

    let n = (*inputs).data.len() / 3;
    let mut xs = ArrayTrait::new();
    let mut vs = ArrayTrait::new();

    let mut i = 0;
    while i != n {
        let (x, v) = run_momentum(
            r,
            t,
            *(*inputs).data.at(i),
            *(*inputs).data.at(n + i),
            *(*inputs).data.at(n * 2 + i),
            alpha,
            beta,
            mode,
            norm_coefficient
        );
        xs.append(x);
        vs.append(v);
        i += 1;
    };
    return (
        TensorTrait::new(array![xs.len()].span(), xs.span()),
        TensorTrait::new(array![vs.len()].span(), vs.span())
    );
}


fn run_momentum<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +Sub<T>,
    +PartialOrd<T>,
>(
    r: T, t: T, x: T, g: T, v: T, alpha: T, beta: T, mode: MODE, norm_coefficient: T,
) -> (T, T) {
    match mode {
        MODE::STANDARD => { apply_momentum(r, t, x, g, v, alpha, beta, mode, norm_coefficient) },
        MODE::NESTEROV => { apply_nesterov(r, t, x, g, v, alpha, beta, mode, norm_coefficient) }
    }
}

fn apply_momentum<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +Sub<T>,
    +PartialOrd<T>,
>(
    r: T, t: T, x: T, g: T, v: T, alpha: T, beta: T, mode: MODE, norm_coefficient: T,
) -> (T, T) {
    let g_regularized = norm_coefficient * x + g;

    let beta_adjusted = if t > NumberTrait::zero() {
        beta
    } else {
        NumberTrait::one()
    };
    let v_new = alpha * v + beta_adjusted * g_regularized;
    let x_new = x - r * v_new;

    return (x_new, v_new);
}

fn apply_nesterov<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +Sub<T>,
    +PartialOrd<T>,
>(
    r: T, t: T, x: T, g: T, v: T, alpha: T, beta: T, mode: MODE, norm_coefficient: T,
) -> (T, T) {
    let g_regularized = norm_coefficient * x + g;

    let beta_adjusted = if t > NumberTrait::zero() {
        beta
    } else {
        NumberTrait::one()
    };
    let v_new = alpha * v + beta_adjusted * g_regularized;
    let x_new = x - r * (g_regularized + alpha * v_new);

    return (x_new, v_new);
}

