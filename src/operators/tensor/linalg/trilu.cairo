use array::ArrayTrait;
use array::SpanTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

/// Cf: TensorTrait::trilu docstring
fn trilu<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, upper: bool, k: i64
) -> Tensor<T> {
    assert((*self.shape).len() >= 2, 'must have at least 2 dimensions');

    let shape_len = (*self.shape).len();
    let mut output_data = ArrayTrait::new();

    let mut batch_size = 1;
    let n: u32 = *self.shape[shape_len - 2];
    let m: u32 = *self.shape[shape_len - 1];

    {
        let mut i = 0;
        loop {
            if i == shape_len - 2 {
                break ();
            }
            batch_size *= *self.shape[i];
            i += 1;
        }
    }

    {
        let mut b = 0;
        loop {
            if b == batch_size {
                break ();
            }

            let mut i = 0;
            loop {
                if i == n {
                    break ();
                }
                let mut j = 0;
                loop {
                    if j == m {
                        break ();
                    }

                    let ii: felt252 = i.into();
                    let jj: felt252 = j.into();

                    let iii: i64 = ii.try_into().unwrap();
                    let jjj: i64 = jj.try_into().unwrap();

                    let result = if (upper && (iii + k <= jjj)) || (!upper && (iii + k >= jjj)) {
                        *(*self.data)[(b * n * m) + (i * m + j)]
                    } else {
                        NumberTrait::zero()
                    };
                    output_data.append(result);
                    j += 1;
                };
                i += 1;
            };
            b += 1;
        };
    }

    return TensorTrait::new(*self.shape, output_data.span());
}
