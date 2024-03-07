use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::core::{stride, unravel_index};
use orion::operators::tensor::math::max_in_tensor::max_in_tensor;
use orion::operators::tensor::math::min_in_tensor::min_in_tensor;


/// Computes the Cartesian product of multiple arrays.
///
/// # Arguments
/// * `arrays` - `Span<Span<usize>>`, Span containing N spans of usize elements.
///
/// # Example
// cartesian([1, 2, 3], [4, 5], [6, 7])
///
/// >>> [
///         [1, 4, 6],
///         [1, 4, 7],
///         [1, 5, 6],
///         [1, 5, 7],
///         [2, 4, 6],
///         [2, 4, 7],
///         [2, 5, 6],
///         [2, 5, 7],
///         [3, 4, 6],
///         [3, 4, 7],
///         [3, 5, 6],
///         [3, 5, 7]
///      ]
///
/// # Returns
/// * A `Span<Span<usize>>` containing the result of the Cartesian product.
fn cartesian(mut arrays: Span<Span<usize>>,) -> Span<Span<usize>> {
    let n_array = arrays.len();
    let mut res = ArrayTrait::new();
    let mut n_item = 1;
    let mut size_arrays = ArrayTrait::new();
    let mut iter = arrays.clone();
    loop {
        match iter.pop_front() {
            Option::Some(array) => {
                let dim = (*array).len();
                n_item *= dim;
                size_arrays.append(dim);
            },
            Option::None => { break; }
        }
    };
    let stride = stride(size_arrays.span());

    let mut i = 0;
    loop {
        if i == n_item {
            break;
        }
        let mut flatten_index = i;
        let mut item = ArrayTrait::new();

        let mut n = 0;
        loop {
            if n == n_array {
                break;
            }
            let (n_index, rem) = DivRem::div_rem(
                flatten_index, (*stride.at(n)).try_into().unwrap()
            );
            flatten_index = rem;
            item.append(*(*arrays.at(n)).at(n_index));
            n += 1;
        };
        res.append(item.span());
        i += 1;
    };

    return res.span();
}


/// Computes all coordinates given the shape of a tensor.
///
/// # Arguments
/// * `shape` - `Span<usize>`, A span containing the shape of the tensor as usize elements.
///
/// # Returns
/// * A span of spans representing all possible coordinates of the tensor.
fn get_all_coord(mut shape: Span<usize>) -> Span<Span<usize>> {
    let mut res = ArrayTrait::new();

    let stride = stride(shape);
    let n_item = *stride.at(0) * *shape.at(0);
    let dim = shape.len();

    let mut i = 0;
    loop {
        if i == n_item {
            break;
        }
        let mut flatten_index = i;
        let mut indices = ArrayTrait::new();

        let mut n = 0;
        loop {
            if n == dim {
                break;
            }
            let (n_index, rem) = DivRem::div_rem(
                flatten_index, (*stride.at(n)).try_into().unwrap()
            );
            flatten_index = rem;
            indices.append(n_index);
            n += 1;
        };
        res.append(indices.span());
        i += 1;
    };

    return res.span();
}

/// Checks if an index is out of bounds given the shape of a tensor.
///
/// # Arguments
/// * `ind` - `Span<usize>` - A span containing the index of the tensor as usize elements.
/// * `shape` - `Span<usize>` - A span containing the shape of the tensor as usize elements.
///
/// # Returns
/// * `true` if the index is out of bounds, otherwise `false`.
fn is_out(ind: Span<usize>, shape: Span<usize>,) -> bool {
    let mut n = 0;
    let is_out = loop {
        if n == ind.len() {
            break false;
        }
        let s = *shape.at(n);
        let i = *ind.at(n);
        if i < 0 {
            break true;
        }
        if i >= s {
            break true;
        }
        n += 1;
    };
    return is_out;
}

/// Computes the product of all the elements of the input span
///
/// # Arguments
/// * `a` - `Span<T>`, input span.
///
/// # Returns
/// * `prod` - `T`, result of the product.
fn prod<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TensorTrait<T>, +MulEq<T>,>(
    mut a: Span<T>
) -> T {
    let mut prod = NumberTrait::one();
    loop {
        match a.pop_front() {
            Option::Some(v) => { prod *= *v; },
            Option::None => { break prod; }
        };
    }
}


/// Computes the product of all the elements of the input span
///
/// # Arguments
/// * `a` - `Span<T>`, input span.
/// * `start` - usize.
///
/// # Returns
/// * `prod` - `T`, result of the product.
fn prod_on_subset<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TensorTrait<T>, +Mul<T>,>(
    pA: Span<T>, start: usize
) -> T {
    let mut i = start;
    let mut prod = NumberTrait::one();
    while i != pA.len() {
        prod = prod * (*pA.at(i));
        i += 1;
    };

    prod
}

/// Computes the dot product of the inputs span
///
/// # Arguments
/// * `a` - `Span<T>`, input span.
/// * `b` - `Span<T>`, input span.
///
/// # Returns
/// * `acc` - `T`, result of the dot product.
fn dot<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TensorTrait<T>, +AddEq<T>, +Mul<T>,>(
    a: Span<T>, b: Span<T>
) -> T {
    let mut i = 0;
    let mut acc = NumberTrait::zero();
    while i != a.len() {
        acc += *a.at(i) * *b.at(i);
        i += 1;
    };

    acc
}


/// Return evenly spaced values within a given interval.
/// Values are generated within the half-open interval [0, end) (in other words, the interval including start but excluding stop).
///
/// # Arguments
/// * `start` - usize
/// * `end` - usize
/// * `step` - usize
///
/// # Returns
//// returns a span of len ceil((end - start) / step), containing the values from `start` to the closest integer to `end` in the interval [0, end) with interval `step`.
fn arange(start: usize, end: usize, step: usize) -> Span<usize> {
    let mut arr: Array<usize> = array![];
    let mut i = start;
    while i < end {
        arr.append(i);
        i += step;
    };

    arr.span()
}

/// Return a span containing `n` zeros of type `T`
///
/// # Arguments
/// * `n` - usize
///
fn zeros<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>>(n: usize) -> Span<T> {
    let mut zeros: Array<T> = array![];
    let mut i = 0;
    while i != n {
        zeros.append(NumberTrait::zero());
        i += 1;
    };

    zeros.span()
}


/// Round elements of the span to the nearest integer. For values exactly halfway between rounded decimal valuesrounds to the nearest even value. 
///
/// # Arguments
/// * `data` - `Span<T>` 
///
/// # Returns
//// a `Span<T>` countaining the rounded values.
fn rint<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +SubEq<T>,
    +Rem<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
    +Add<T>,
    +Sub<T>
>(
    data: Span<T>
) -> Span<T> {
    let mut rint: Array<T> = array![];
    let two: T = NumberTrait::one() + NumberTrait::one();

    let mut i = 0;
    while i != data.len() {
        let x = *data.at(i);
        let mut round = NumberTrait::round(x);

        let diff = round - x;
        if diff == NumberTrait::half() {
            if round % two != NumberTrait::zero() {
                round -= NumberTrait::one()
            }
        }

        rint.append(round);
        i += 1;
    };

    rint.span()
}


/// Reverse the span input
///
/// # Arguments
/// * `data` - `Span<T>` 
///
/// # Returns
//// a `Span<T>` countaining the reversed values.
fn reverse<T, +Copy<T>, +Drop<T>,>(data: Span<T>) -> Span<T> {
    let mut rev: Array<T> = array![];
    let mut i = data.len();
    while i != 0 {
        rev.append(*data.at(i - 1));
        i -= 1;
    };

    rev.span()
}

