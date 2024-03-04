use core::array::ArrayTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor,};
use orion::operators::vec::{NullableVec, NullableVecImpl};
use orion::operators::tensor::core::{stride};
use core::debug::PrintTrait;
use core::traits::Into;
use orion::numbers::{U32IntoI32, I32IntoU32, I32Div, I32Number};


use orion::operators::nn::functional::grid_sample::{grid_sample};


/// Cf: NNTrait::deform_conv docstring
fn deform_conv<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +Sub<T>,
    +Div<T>,
    +AddEq<T>,
    +PrintTrait<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +Rem<T>,
    +Neg<T>,
    +SubEq<T>,
    +Mul<Tensor<T>>,
>(
    X: @Tensor<T>,
    W: @Tensor<T>,
    offset: @Tensor<T>,
    B: Option<Span<T>>,
    mask: Option<Tensor<T>>,
    dilations: Option<Span<usize>>,
    group: Option<usize>,
    kernel_shape: Option<Span<usize>>,
    offset_group: Option<usize>,
    pads: Option<Span<usize>>,
    strides: Option<Span<usize>>,
) -> Tensor<T> {
    assert((*X).shape.len() >= 3, 'X must have at least 3 dim');
    assert((*W).shape.len() >= 3, 'X must have at least 3 dim');

    let dilations = match dilations {
        Option::Some(dilations) => dilations,
        Option::None => {
            let mut dilations = ArrayTrait::new();
            let mut i = 2;
            while i != (*X).shape.len() {
                dilations.append(1);
                i += 1;
            };
            dilations.span()
        },
    };
    let kernel_shape = match kernel_shape {
        Option::Some(kernel_shape) => kernel_shape,
        Option::None => {
            let mut kernel_shape = ArrayTrait::new();
            let mut i = 2;
            while i != (*W).shape.len() {
                kernel_shape.append(*(*W).shape.at(i));
                i += 1;
            };
            kernel_shape.span()
        },
    };
    let pads = match pads {
        Option::Some(pads) => pads,
        Option::None => {
            let mut pads = ArrayTrait::new();
            let mut i = 2;
            while i != (*X).shape.len() {
                pads.append(0);
                pads.append(0);
                i += 1;
            };
            pads.span()
        },
    };
    let strides = match strides {
        Option::Some(strides) => strides,
        Option::None => {
            let mut strides = ArrayTrait::new();
            let mut i = 2;
            while i != (*X).shape.len() {
                strides.append(1);
                i += 1;
            };
            strides.span()
        },
    };
    let group = match group {
        Option::Some(group) => group,
        Option::None => { 1 },
    };

    let offset_group = match offset_group {
        Option::Some(offset_group) => offset_group,
        Option::None => { 1 },
    };

    let n = *(*X).shape.at(0);
    let ic = *(*X).shape.at(1);
    let oc = *(*W).shape.at(0);
    let output_shape = SpanTrait::slice((*offset).shape, 2, (*offset).shape.len() - 2);

    assert(ic == *(*W).shape.at(1) * group, 'shape inconsistencies');
    assert(oc % group == 0, 'shape inconsistencies');

    let ics_per_group = *(*W).shape.at(1);
    let ocs_per_group = oc / group;

    assert(ic % offset_group == 0, 'offset_group inconsistencies');

    let ics_per_offset_group = ic / offset_group;

    assert(
        offset_group * prod(kernel_shape, 0) * kernel_shape.len() == *(*offset).shape.at(1),
        'offset_group inconsistencies'
    );

    let mut offset_shape = array![n, offset_group];
    offset_shape.append_span(kernel_shape);
    offset_shape.append(kernel_shape.len());
    offset_shape.append_span(output_shape);

    let offset = offset.reshape(offset_shape.span());

    let mask = match mask {
        Option::Some(mask) => mask,
        Option::None => {
            let mut mask = ArrayTrait::<T>::new();
            let mask_end = n * offset_group * prod(kernel_shape, 0) * prod(output_shape, 0);
            let mut i = 0;
            while i != mask_end {
                mask.append(NumberTrait::<T>::one());
                i += 1;
            };
            let mut mask_shape = array![n, offset_group * prod(kernel_shape, 0)];
            mask_shape.append_span(output_shape);
            TensorTrait::new(mask_shape.span(), mask.span())
        },
    };

    let mut mask_shape = array![n, offset_group];
    mask_shape.append_span(kernel_shape);
    mask_shape.append_span(output_shape);
    let mask = mask.reshape(mask_shape.span());

    if (*X).shape.len() == 4 {
        let ih: T = NumberTrait::new_unscaled((*(*X).shape.at(2)).into(), false);
        let iw: T = NumberTrait::new_unscaled((*(*X).shape.at(3)).into(), false);

        let x_stride = stride((*X).shape);
        let w_stride = stride((*W).shape);
        let offset_stride = stride(offset.shape);
        let mask_stride = stride(mask.shape);

        let mut x_subset_shape = array![1, 1];
        x_subset_shape.append_span(SpanTrait::slice(*(X).shape, 2, (*(X).shape).len() - 2));
        let x_subset_shape = x_subset_shape.span();

        let mut w_subset_shape = array![1, 1];
        w_subset_shape.append_span(SpanTrait::slice(*(W).shape, 2, (*(W).shape).len() - 2));
        let w_subset_shape = w_subset_shape.span();

        let oh = *offset.shape.at(offset_shape.len() - 2);
        let ow = *offset.shape.at(offset_shape.len() - 1);

        let kh = *kernel_shape.at(0);
        let kw = *kernel_shape.at(1);

        let sth: T = NumberTrait::new_unscaled((*strides.at(0)).into(), false);
        let stw: T = NumberTrait::new_unscaled((*strides.at(1)).into(), false);

        let dh = *dilations.at(0);
        let dw = *dilations.at(1);

        let kh_new = (kh - 1) * dh + 1;
        let kw_new = (kw - 1) * dw + 1;

        let bh: T = NumberTrait::new_unscaled((*pads.at(0)).into(), true);
        let bw: T = NumberTrait::new_unscaled((*pads.at(1)).into(), true);

        assert(
            oh == (((*(*X).shape.at(2) - kh_new + *pads.at(0) + *pads.at(2)) / *strides.at(0)) + 1),
            'incompatible shapes'
        );
        assert(
            ow == (((*(*X).shape.at(3) - kw_new + *pads.at(1) + *pads.at(3)) / *strides.at(1)) + 1),
            'incompatible shapes'
        );

        let mut res = NullableVecImpl::new();
        let res_shape = array![n, oc, oh, ow].span();
        let res_stride = stride(res_shape);
        res.set(n * *res_stride.at(0) - 1, NumberTrait::zero());

        match B {
            Option::Some(B) => {
                let mut i = 0;
                while i != n {
                    let mut j = 0;
                    while j != oc {
                        let b_j = *B.at(j);
                        let mut k = 0;
                        while k != oh {
                            let mut l = 0;
                            while l != ow {
                                res
                                    .set(
                                        i * *res_stride.at(0)
                                            + j * *res_stride.at(1)
                                            + k * *res_stride.at(2)
                                            + l,
                                        b_j
                                    );
                                l += 1;
                            };
                            k += 1;
                        };
                        j += 1;
                    };
                    i += 1;
                };
            },
            Option::None => {},
        }

        let (kernel_pos_w, kernel_pos_h) = meshgrid(arange(0, kw_new, dw), arange(0, kh_new, dh));
        let kernel_pos_wrt_first_elem = stack(kernel_pos_h, kernel_pos_w);

        let dh: T = NumberTrait::new_unscaled(dh.into(), false);
        let dw: T = NumberTrait::new_unscaled(dw.into(), false);

        let kh_new: T = NumberTrait::new_unscaled(kh_new.into(), false);
        let kw_new: T = NumberTrait::new_unscaled(kw_new.into(), false);

        // dimension of kernel_pos_wrt_first_elem is ks0 x ks1
        let ks0 = NumberTrait::ceil(kh_new / dh).try_into().unwrap();
        let ks1 = NumberTrait::ceil(kw_new / dw).try_into().unwrap();

        let one: T = NumberTrait::one();
        let two: T = NumberTrait::one() + NumberTrait::one();

        let mut batch_idx = 0;
        while batch_idx != n {
            let mut oc_idx = 0;
            while oc_idx != oc {
                let mut ic_idx = 0;
                while ic_idx != ic {
                    if (ic_idx / ics_per_group) == (oc_idx / ocs_per_group) {
                        let offset_group_idx = ic_idx / ics_per_offset_group;

                        let mut i = 0;
                        while i != oh {
                            let index = NumberTrait::new_unscaled(i.into(), false);
                            let h_coord = bh + sth * index;
                            let mut j = 0;
                            while j != ow {
                                let jndex = NumberTrait::new_unscaled(j.into(), false);
                                let w_coord = bw + stw * jndex;

                                let mut kernel = copy_to_vec(kernel_pos_wrt_first_elem);
                                let mut mask_subset = ArrayTrait::new();
                                let mut kernel_test = ArrayTrait::new();
                                let mut offset_TEST = ArrayTrait::new();

                                let mut hi = 0;
                                while hi != ks0 {
                                    let mut wi = 0;
                                    while wi != ks1 {
                                        let elem1 = h_coord
                                            + *offset
                                                .data
                                                .at(
                                                    batch_idx * *offset_stride.at(0)
                                                        + offset_group_idx * *offset_stride.at(1)
                                                        + hi * *offset_stride.at(2)
                                                        + wi * *offset_stride.at(3)
                                                        + 0 * *offset_stride.at(4)
                                                        + i * *offset_stride.at(5)
                                                        + j
                                                );
                                        let elem2 = w_coord
                                            + *offset
                                                .data
                                                .at(
                                                    batch_idx * *offset_stride.at(0)
                                                        + offset_group_idx * *offset_stride.at(1)
                                                        + hi * *offset_stride.at(2)
                                                        + wi * *offset_stride.at(3)
                                                        + 1 * *offset_stride.at(4)
                                                        + i * *offset_stride.at(5)
                                                        + j
                                                );

                                        mask_subset
                                            .append(
                                                *mask
                                                    .data
                                                    .at(
                                                        batch_idx * *mask_stride.at(0)
                                                            + offset_group_idx * *mask_stride.at(1)
                                                            + hi * *mask_stride.at(2)
                                                            + wi * *mask_stride.at(3)
                                                            + i * *mask_stride.at(4)
                                                            + j
                                                    )
                                            );
                                        kernel_test.append(kernel.at(hi * (ks1 * 2) + wi * 2));
                                        offset_TEST
                                            .append(
                                                *offset
                                                    .data
                                                    .at(
                                                        batch_idx * *offset_stride.at(0)
                                                            + offset_group_idx
                                                                * *offset_stride.at(1)
                                                            + hi * *offset_stride.at(2)
                                                            + wi * *offset_stride.at(3)
                                                            + 0 * *offset_stride.at(4)
                                                            + i * *offset_stride.at(5)
                                                            + j
                                                    )
                                            );
                                        kernel
                                            .set(
                                                hi * (ks1 * 2) + wi * 2,
                                                (kernel.at(hi * (ks1 * 2) + wi * 2) + elem1)
                                                    / (ih - one)
                                                    * two
                                                    - one
                                            );
                                        kernel
                                            .set(
                                                hi * (ks1 * 2) + wi * 2 + 1,
                                                (kernel.at(hi * (ks1 * 2) + wi * 2 + 1) + elem2)
                                                    / (iw - one)
                                                    * two
                                                    - one
                                            );
                                        wi += 1;
                                    };
                                    hi += 1;
                                };
                                let kernel = flip_mod_2(ref kernel);

                                let subset_x = TensorTrait::new(
                                    x_subset_shape,
                                    SpanTrait::slice(
                                        (*X).data,
                                        batch_idx * *x_stride.at(0) + ic_idx * *x_stride.at(1),
                                        *x_stride.at(1)
                                    )
                                );
                                let subset_w = TensorTrait::new(
                                    w_subset_shape,
                                    SpanTrait::slice(
                                        (*W).data,
                                        oc_idx * *w_stride.at(0)
                                            + (ic_idx % ics_per_group) * *w_stride.at(1),
                                        *w_stride.at(1)
                                    )
                                );
                                let mask_subset = TensorTrait::new(
                                    array![1, 1, ks0, ks1].span(), mask_subset.span()
                                );
                                let kernel = TensorTrait::new(
                                    array![1, ks0, ks1, 2].span(), kernel
                                );

                                let grid_sample_output = grid_sample(
                                    @subset_x, @kernel, Option::Some(1), Option::None, Option::None
                                );

                                // broadcasted multiply
                                let conv_value = (grid_sample_output * subset_w);
                                let conv_value = (conv_value * mask_subset);

                                res
                                    .set(
                                        batch_idx * *res_stride.at(0)
                                            + oc_idx * *res_stride.at(1)
                                            + i * *res_stride.at(2)
                                            + j,
                                        res
                                            .at(
                                                batch_idx * *res_stride.at(0)
                                                    + oc_idx * *res_stride.at(1)
                                                    + i * *res_stride.at(2)
                                                    + j
                                            )
                                            + sum(conv_value.data, 0)
                                    );
                                j += 1;
                            };
                            i += 1;
                        };
                    }
                    ic_idx += 1;
                };
                oc_idx += 1;
            };
            batch_idx += 1;
        };

        let mut res_data = ArrayTrait::new();
        let mut i = 0;
        while i != res.len() {
            res_data.append(res.at(i));
            i += 1;
        };
        return TensorTrait::new(res_shape, res_data.span());
    }

    panic(array!['not supported yet!'])
}


fn meshgrid(x: Span<usize>, y: Span<usize>) -> (Span<usize>, Span<usize>) {
    let mut xv = ArrayTrait::new();
    let mut yv = ArrayTrait::new();

    let mut i = 0;
    while i != y.len() {

        xv.append_span(x);
        let mut j = 0;
        while j != x.len() {
            yv.append(*y.at(i));
            j += 1;
        };
        i += 1;
    };
    return (xv.span(), yv.span());
}

fn stack(x: Span<usize>, y: Span<usize>) -> Span<usize> {
    let mut stack = ArrayTrait::new();

    let mut i = 0;
    while i != x.len() {
        stack.append(*x.at(i));
        stack.append(*y.at(i));
        i += 1;
    };

    return stack.span();
}


fn flip_mod_2<T, MAG, impl TDrop: Drop<T>, impl TCopy: Copy<T>, +NumberTrait<T, MAG>>(
    ref x: NullableVec<T>
) -> Span<T> {
    let mut i = 0;
    let mut res = ArrayTrait::new();
    while i != x.len / 2 {
        res.append(x.at(i * 2 + 1));
        res.append(x.at(i * 2));
        i += 1;
    };

    return res.span();
}

fn copy_to_vec<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TryInto<T, usize>, +Into<usize, MAG>,
>(
    x: Span<usize>
) -> NullableVec<T> {
    let mut res = NullableVecImpl::new();

    let mut i = 0;
    while i != x.len() {
        res.set(i, NumberTrait::new_unscaled((*x.at(i)).into(), false));
        i += 1;
    };

    return res;
}

// return a span of len ceil((end - start) / step)
fn arange(start: usize, end: usize, step: usize) -> Span<usize> {
    let mut arr = ArrayTrait::new();
    let mut i = start;
    while i != end {
        arr.append(i);
        i += step;
    };
    return arr.span();
}


fn prod<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TensorTrait<T>, +Mul<T>,>(
    a: Span<T>, start: usize
) -> T {
    assert(a.len() > start, 'wrong input dim');
    let mut prod = NumberTrait::one();
    let mut i = start;
    while i != a.len() {
        prod = prod * (*a.at(i));
        i += 1;
    };
    return prod;
}



fn sum<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TensorTrait<T>, +AddEq<T>,>(
    a: Span<T>, start: usize
) -> T {
    assert(a.len() > start, 'wrong input dim');
    let mut sum = NumberTrait::zero();
    let mut i = start;
    while i != a.len() {
        sum += (*a.at(i));
        i += 1;
    };
    return sum;
}
