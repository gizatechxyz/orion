use orion::operators::tensor::{Tensor, TensorTrait, U32Tensor};
use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;
use orion::numbers::NumberTrait;
use alexandria_data_structures::array_ext::{ArrayTraitExt, SpanTraitExt};

/// Cf: Tensor::center_crop_pad docstring
fn center_crop_pad<T, +Copy<T>, +Drop<T>, +TensorTrait<T>,>(
    self: @Tensor<T>, shape: Tensor<usize>, axes: Option<Array<i64>>, zero: T
) -> Tensor<T> {
    let input_rank: usize = (*self.shape).len();
    let mut axes: Array<usize> = match axes {
        Option::Some(mut value) => {
            let mut axes: Array<usize> = ArrayTrait::new();
            loop {
                match value.pop_front() {
                    Option::Some(axis) => {
                        if axis >= 0 {
                            axes.append(axis.try_into().unwrap());
                        } else {
                            let mut input_rank: i64 = input_rank.into();
                            assert!(axis + input_rank >= 0, "shape cannot be less than 0");
                            axes.append((axis + input_rank).try_into().unwrap());
                        }
                    },
                    Option::None => { break (); },
                };
            };
            axes
        },
        Option::None => {
            let mut axes: Array<usize> = ArrayTrait::new();
            let mut i: usize = 0;

            while i < input_rank {
                axes.append(i);
                i += 1;
            };
            axes
        }
    };

    let mut pad_slices: Array<Array<usize>> = ArrayTrait::new();
    let mut crop_slices: Array<Array<usize>> = ArrayTrait::new();
    let mut self_shape_copy = (*self.shape).clone();
    loop {
        match self_shape_copy.pop_front() {
            Option::Some(dim) => {
                let mut temp: Array<usize> = ArrayTrait::new();
                let mut i: usize = 0;

                while i < *dim {
                    temp.append(i);
                    i += 1;
                };
                pad_slices.append(temp.clone());
                crop_slices.append(temp.clone());
            },
            Option::None(_) => { break (); }
        };
    };

    let mut new_shape: Array<usize> = ArrayTrait::new();
    let mut self_shape_copy = (*self.shape).clone();
    loop {
        match self_shape_copy.pop_front() {
            Option::Some(dim) => { new_shape.append(*dim); },
            Option::None(_) => { break (); }
        };
    };

    let mut i: usize = 0;
    loop {
        let mut a: usize = match axes.pop_front() {
            Option::Some(axes) => axes.try_into().unwrap(),
            Option::None(_) => { break (); }
        };

        let mut sh: usize = match shape.data.get(i) {
            Option::Some(sh) => {
                let res: usize = (*sh.unbox()).try_into().unwrap();
                res
            },
            Option::None(_) => { break (); }
        };

        let mut dim: usize = (*(*self.shape).at(a));
        if sh == a {
            continue;
        } else if sh < dim {
            usize_cover(ref new_shape, a, sh);
            let mut d = dim - sh;
            let mut sl: Array<usize> = ArrayTrait::new();
            if d % 2 == 0 {
                d /= 2;
                sl = slice(d, dim - d);
            } else {
                d /= 2;
                sl = slice(d, dim - d - 1);
            }
            array_cover(ref crop_slices, a, sl);
        } else {
            // sh > dim
            usize_cover(ref new_shape, a, sh);
            let mut d = sh - dim;
            let mut sl: Array<usize> = ArrayTrait::new();
            if d % 2 == 0 {
                d /= 2;
                sl = slice(d, sh - d);
            } else {
                d /= 2;
                sl = slice(d, sh - d - 1);
            }
            array_cover(ref pad_slices, a, sl);
        };
        i += 1;
    };

    let mut cropped = tensor_crop(self, crop_slices);
    let result = tensor_pad(cropped, pad_slices, new_shape, zero);
    result
}

fn tensor_pad<T, +Copy<T>, +Drop<T>, +TensorTrait<T>,>(
    input_data: Tensor<T>, mut pad_slices: Array<Array<usize>>, shape: Array<usize>, zero: T
) -> Tensor<T> {
    let mut count: usize = 1;
    let mut res: Span<T> = input_data.data;
    let mut shape_copy = shape.clone();
    let mut i: usize = input_data.shape.len().into() - 1;
    loop {
        let mut shape_i = shape_copy.at(i);
        let mut input_data_shape_i = input_data.shape.at(i);
        let mut slice = pad_slices.at(i);
        let mut slice_len = slice.len();
        if slice_len > *shape_i {
            slice_len = *shape_i;
        }
        if i == 0 {
            if shape_i != input_data_shape_i {
                let mut temp = res;
                res = ArrayTrait::<T>::new().span();
                res = make_zero_array(*slice.at(0) * count, zero)
                    .concat(temp)
                    .concat(
                        make_zero_array((*shape_i - *slice.at(slice_len - 1) - 1) * count, zero)
                    );
            }
            break ();
        }
        if shape_i != input_data_shape_i {
            let mut arr_list: Array<Array<T>> = make_array_from_dim(
                res, count * *input_data_shape_i
            );
            res = ArrayTrait::<T>::new().span();
            loop {
                match arr_list.pop_front() {
                    Option::Some(mut arr) => {
                        res = res.concat(make_zero_array(*slice.at(0) * count, zero));
                        res = res.concat(arr.span());
                        res = res
                            .concat(
                                make_zero_array(
                                    (*shape_i - *slice.at(slice_len - 1) - 1) * count, zero
                                )
                            );
                    },
                    Option::None(_) => { break (); }
                };
            };
        }
        count *= *shape_i;
        i -= 1;
    };
    TensorTrait::<T>::new(shape.span(), res)
}

fn tensor_crop<T, +Copy<T>, +Drop<T>, +TensorTrait<T>,>(
    input_data: @Tensor<T>, mut crop_slices: Array<Array<usize>>
) -> Tensor<T> {
    let mut input_data_shape_copy: Span<usize> = *input_data.shape;
    let mut count = 1;
    let mut shape: Array<usize> = ArrayTrait::new();
    let mut i: usize = 0;

    while i < input_data_shape_copy.len() {
        shape.append(*input_data_shape_copy.at(i));
        i += 1;
    };

    let mut res: Span<T> = *input_data.data;
    let mut i: usize = input_data_shape_copy.len() - 1;
    loop {
        let mut dim = (*input_data_shape_copy.at(i));
        let mut slice = crop_slices.at(i);
        let slice_len: usize = slice.len();
        if i == 0 {
            if dim != slice_len {
                usize_cover(ref shape, i, slice_len);
                let mut arr_list: Array<Array<T>> = make_array_from_dim(res, count);
                res = ArrayTrait::<T>::new().span();
                let mut j: usize = 0;

                while j < slice_len {
                    res = res.concat(arr_list.at(*slice.at(j)).span());
                    j += 1;
                };
            }
            break ();
        }

        if dim != slice_len {
            usize_cover(ref shape, i, slice_len);
            let mut arr_list: Array<Array<T>> = make_array_from_dim(res, count * dim);
            res = ArrayTrait::<T>::new().span();
            loop {
                match arr_list.pop_front() {
                    Option::Some(mut arr) => {
                        let mut arr = make_array_from_dim(arr.span(), count);
                        let mut j: usize = 0;

                        while j < slice_len {
                            res = res.concat(arr.at(*slice.at(j)).span());
                            j += 1;
                        };
                    },
                    Option::None(_) => { break (); }
                };
            };
        }
        count *= slice_len;
        i -= 1;
    };
    TensorTrait::new(shape.span(), res)
}

fn make_zero_array<T, +Drop<T>, +Copy<T>>(size: usize, zero: T) -> Span<T> {
    let mut res: Array<T> = ArrayTrait::new();
    let mut i: usize = 0;

    while i < size {
        res.append(zero.clone());
        i += 1;
    };
    res.span()
}

fn slice(start: usize, end: usize) -> Array<usize> {
    let mut index: Array<usize> = ArrayTrait::new();
    let mut i: usize = start;

    while i < end {
        index.append(i);
        i += 1;
    };
    index
}

fn array_cover(ref arr: Array<Array<usize>>, index: usize, data: Array<usize>) {
    if arr.is_empty() {
        arr.append(data);
        return ();
    }

    let mut arr_len: usize = arr.len();
    let mut i: usize = 0;

    while i < arr_len {
        let temp = arr.pop_front().unwrap();
        if i == index {
            arr.append(data.clone());
        } else {
            arr.append(temp);
        }
        i += 1;
    };
}

fn usize_cover(ref arr: Array<usize>, index: usize, data: usize) {
    if arr.is_empty() {
        arr.append(data);
        return ();
    }

    let mut arr_len: usize = arr.len();
    let mut i: usize = 0;

    while i < arr_len {
        let temp = arr.pop_front().unwrap();
        if i == index {
            arr.append(data.clone());
        } else {
            arr.append(temp);
        }
        i += 1;
    };
}

fn make_array_from_dim<T, +Drop<T>, +Copy<T>>(input_data: Span<T>, dim: usize) -> Array<Array<T>> {
    let row: usize = input_data.len() / dim;
    let data_copy: Span<T> = input_data.clone();

    let mut res = ArrayTrait::<Array<T>>::new();
    let mut i: usize = 0;

    while i < row {
        let mut temp: Array<T> = ArrayTrait::new();
        let mut j: usize = 0;
        loop {
            if j > dim - 1 {
                break ();
            }
            temp.append((*data_copy.at(i * dim + j)));
            j += 1;
        };
        res.append(temp);
        i += 1;
    };
    res
}
