use core::option::OptionTrait;
use orion::operators::tensor::{Tensor, TensorTrait, U32Tensor};
use orion::operators::matrix::{MutMatrixTrait, MutMatrix, MutMatrixImpl};

/// Cf: NNTrait::split docstring
fn split_to_sequence<T, +Copy<T>, +Drop<T>, +TensorTrait<T>,>(
    self: @Tensor<T>, axis: usize, keepdims: usize, split: Option<Tensor<usize>>
) -> Array<Tensor<T>> {
    let has_split = match split {
        Option::Some => true,
        Option::None => false,
    };
    let mut has_num_outputs = false;
    let mut split_unwrap: Tensor<usize> = TensorTrait::new(array![1].span(), array![1].span());

    if (!has_split) {
        let split_length = *(*self.shape).at(axis);
        let mut split_data: Array<usize> = array![];
        let mut i = 0;
        while i != split_length {
            split_data.append(1);
            i += 1;
        };

        split_unwrap = TensorTrait::new(array![split_length].span(), split_data.span());
    } else if (split.unwrap().data.len() == 1 && *(split.unwrap().shape.at(0)) == 1) {
        // A scalar
        has_num_outputs = true;
        split_unwrap = split.unwrap();
    } else {
        split_unwrap = split.unwrap();
    }

    let mut splited_t: Array<Tensor<T>> = array![];

    let rank = (*self).shape.len();
    // assert(axis < rank && axis > -rank, 'axis out of dimensions');
    assert(axis < rank, 'axis out of dimensions');

    if (has_num_outputs) {
        splited_t = split_num_outputs(self, axis, *(split_unwrap.data).at(0));
    } else {
        splited_t = split_has_split(self, axis, split_unwrap);
    }

    if (keepdims == 0 && !has_split) {
        let mut splited_t_temp: Array<Tensor<T>> = array![];
        let mut i = 0;
        while i != splited_t
            .len() {
                let mut shape: Array<i32> = array![];
                let mut j = 0;
                let shape_in_splited: Span<usize> = *splited_t.at(i).shape;
                while j != shape_in_splited
                    .len() {
                        if (j != axis) {
                            shape.append((*shape_in_splited.at(j)).try_into().unwrap())
                        }

                        j += 1;
                    };

                splited_t_temp.append(splited_t[i].reshape(shape.span(), false));
                i += 1;
            };

        return splited_t_temp;
    }
    splited_t
}


/// Subfunction split for tensors (wth num_outputs).
/// Cf: TensorTrait::split docstring
fn split_num_outputs<T, +Copy<T>, +Drop<T>, +TensorTrait<T>,>(
    t: @Tensor<T>, mut axis: usize, num_outputs: usize
) -> Array<Tensor<T>> {
    let mut splited_t: Array<Tensor<T>> = array![];
    let mut div: usize = 0;
    // consturct split array
    let mut split: Array<usize> = array![];
    // if axis==0 {
    //     axis = 1;
    // }
    if (*(*t).shape.at(axis) % num_outputs == 0) {
        div = *(*t).shape.at(axis) / num_outputs;
        let mut i = 0;
        while i != num_outputs {
            split.append(div);
            i += 1;
        };
    } else {
        div = *(*t).shape.at(axis) / num_outputs + 1;
        let mut i = 0;
        while i != num_outputs {
            split.append(div);
            i += 1;
        };

        match split.pop_front() {
            Option::Some(split_last_one) => {
                split.append(split_last_one + *(*t).shape.at(axis) - div * (num_outputs - 1));
            },
            Option::None => { assert(false, 'split is none array'); }
        }
    }

    let mut sli: MutMatrix<usize> = MutMatrixImpl::new((*t).shape.len(), 2);
    let mut pos: usize = 0;
    let mut i = 0;
    while i != (*t)
        .shape
        .len() {
            let s: usize = *(*t).shape.at(i);
            sli.set(i, 0, 0);
            sli.set(i, 1, s);
            i += 1;
        };

    let mut i: usize = 0;
    while i != split
        .len() {
            let spl = *split.at(i);
            sli.set(axis, 0, pos);
            pos += spl;
            sli.set(axis, 1, pos);

            let end_ele_0 = match sli.get(axis, 0) {
                Option::Some(res) => res,
                Option::None => {
                    assert(false, 'Get end_ele_0 is failed');
                    0
                },
            };
            let end_ele_1 = match sli.get(axis, 1) {
                Option::Some(res) => res,
                Option::None => {
                    assert(false, 'Get end_ele_0 is failed');
                    0
                },
            };
            let starts: Span<usize> = array![sli.get(0, 0).unwrap(), end_ele_0].span();
            let ends: Span<usize> = array![sli.get(0, 1).unwrap(), end_ele_1].span();
            let axes: Option<Span<usize>> = Option::None(());
            let steps: Option<Span<usize>> = Option::None(());
            let sub_t: Tensor<T> = t.slice(starts, ends, axes, steps);
            splited_t.append(sub_t);
            i += 1;
        };

    splited_t
}

/// Subfunction split for tensors (wth split).
/// Cf: TensorTrait::split docstring
fn split_has_split<T, +Copy<T>, +Drop<T>, +TensorTrait<T>,>(
    t: @Tensor<T>, axis: usize, split: Tensor<u32>
) -> Array<Tensor<T>> {
    let mut splited_t: Array<Tensor<T>> = array![];
    let mut sli: MutMatrix<usize> = MutMatrixImpl::new((*t).shape.len(), 2);
    let mut pos: usize = 0;
    let mut i = 0;
    while i != (*t)
        .shape
        .len() {
            let s: usize = *(*t).shape.at(i);
            sli.set(i, 0, 0);
            sli.set(i, 1, s);
            i += 1;
        };

    let mut i: usize = 0;
    while i != split
        .data
        .len() {
            let spl: usize = split.at(indices: array![i].span());
            sli.set(axis, 0, pos);
            pos += spl;
            sli.set(axis, 1, pos);

            let end_ele_0 = match sli.get(axis, 0) {
                Option::Some(res) => { res },
                Option::None => {
                    assert(false, 'Get end_ele_0 is failed');
                    0
                },
            };
            let end_ele_1 = match sli.get(axis, 1) {
                Option::Some(res) => { res },
                Option::None => {
                    assert(false, 'Get end_ele_0 is failed');
                    0
                },
            };
            let starts: Span<usize> = array![sli.get(0, 0).unwrap(), end_ele_0].span();
            let ends: Span<usize> = array![sli.get(0, 1).unwrap(), end_ele_1].span();
            let axes: Option<Span<usize>> = Option::None(());
            let steps: Option<Span<usize>> = Option::None(());
            let sub_t: Tensor<T> = t.slice(starts, ends, axes, steps);
            splited_t.append(sub_t);
            i += 1;
        };

    splited_t
}
