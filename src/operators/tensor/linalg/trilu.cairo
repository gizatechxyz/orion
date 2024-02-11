use core::array::ArrayTrait;
use core::array::SpanTrait;

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
    let mut output_size = ArrayTrait::new();

    let mut batch_size = 1;
    let mut n: u32 = 0;
    let mut m: u32 = 0;

    {
        let mut self_shape = *self.shape;
        let mut i = 0;
        loop {
            match self_shape.pop_front() {
                Option::Some(val) => {
                    if i == shape_len - 2 {
                        n = *val;
                    } else if i == shape_len - 1 {
                        m = *val;
                    } else {
                        batch_size *= *val;
                    }
                    i += 1;
                    output_size.append(*val);
                },
                Option::None => { break (); }
            }
        }
    }

    {
        let mut self_data = *self.data;
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

                    let result = match self_data.pop_front() {
                        Option::Some(val) => {
                            if (upper && (iii + k <= jjj)) || (!upper && (iii + k >= jjj)) {
                                *val
                            } else {
                                NumberTrait::zero()
                            }
                        },
                        Option::None => { break (); }
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
