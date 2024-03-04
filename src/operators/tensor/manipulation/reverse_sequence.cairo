use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};

/// Cf: TensorTrait::reverse_sequence docstring
fn reverse_sequence<T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    self: @Tensor<T>,
    sequence_lens: Tensor<usize>,
    batch_axis: Option<usize>,
    time_axis: Option<usize>
) -> Tensor<T> {
    let shape = *self.shape;
    let mut data: Array<T> = array![];

    let has_batch_axis: usize = match batch_axis {
        Option::Some(value) => {
            assert!((value != 0) || (value != 1), "batch_axis must be one of 1 or 0.");
            value
        },
        Option::None => 0,
    };
    let has_time_axis: usize = match time_axis {
        Option::Some(value) => {
            assert!((value != 0) || (value != 1), "time_axis must be one of 1 or 0.");
            value
        },
        Option::None => 1,
    };
    assert!(has_batch_axis != has_time_axis, "batch_axis and time_axis cannot be equal");
    assert!((*self.data).len() >= 2, "Tensor of rank r >= 2");
    let control: bool = if has_batch_axis == 0 && has_time_axis == 1 {
        true
    } else {
        false
    };

    let mut index: Array<usize> = reverse_index(*self.shape, sequence_lens, control);
    loop {
        match index.pop_front() {
            Option::Some(ele) => { data.append(*((*self).data).at(ele)); },
            Option::None => { break; }
        }
    };

    TensorTrait::<T>::new(shape, data.span())
}


fn reverse_index(shape: Span<usize>, sequence_lens: Tensor<usize>, control: bool) -> Array<usize> {
    let x: usize = *shape.at(0);
    let y: usize = *shape.at(1);
    let mut result = ArrayTrait::<usize>::new();

    if control {
        // [i, slice]
        assert!(
            sequence_lens.data.len() <= x, "The length of sequence_lens cannot exceed batch_axis"
        );
        let mut i: usize = 0;
        loop {
            if i >= x {
                break;
            }

            let reverse: usize = (*sequence_lens.data.at(i));
            assert!(
                reverse <= y && reverse >= 1,
                "sequence_lens must be greater than one and less than batch_size"
            );
            let mut j: usize = reverse - 1;
            loop {
                if j == 0 {
                    result.append(i * y + j);
                    break;
                }
                result.append(i * y + j);
                j -= 1;
            };
            let current_index_len: usize = (i + 1) * y - 1;
            let mut j: usize = result.len();
            loop {
                if j > current_index_len {
                    break;
                }
                result.append(j);
                j += 1;
            };
            i += 1;
        };
    } else {
        // [slice, i]
        assert!(
            sequence_lens.data.len() <= y, "The length of sequence_lens cannot exceed time_axis"
        );
        let mut tmp = ArrayTrait::<usize>::new();
        let mut i: usize = 0;
        loop {
            if i > y - 1 {
                break;
            }
            let reverse: usize = *sequence_lens.data.at(i);
            assert!(
                reverse <= x && reverse >= 1,
                "sequence_lens must be greater than one and less than batch_size"
            );

            let mut j: usize = reverse - 1;
            loop {
                if j == 0 {
                    tmp.append(j * y + i);
                    break;
                }
                tmp.append(j * y + i);
                j -= 1;
            };
            let mut j: usize = reverse;
            loop {
                if j > x - 1 {
                    break;
                }
                tmp.append(j * y + i);
                j += 1;
            };
            i += 1;
        };
        let tmp = tmp.span();
        let mut i: usize = 0;
        loop {
            if i > x - 1 {
                break;
            }
            let mut j: usize = 0;
            loop {
                if j > y - 1 {
                    break;
                }
                result.append((*tmp.at(j * x + i)));
                j += 1;
            };
            i += 1;
        };
    }
    result
}
