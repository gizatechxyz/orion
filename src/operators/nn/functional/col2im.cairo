use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{stride};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor,};
use orion::operators::vec::{NullableVec, NullableVecImpl};

fn col2im<T, MAG, +TensorTrait<T>, +NumberTrait<T, MAG>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>,>(
    data: @Tensor<T>,
    image_shape: Span<usize>,
    block_shape: Span<usize>,
    dilations: Option<Span<usize>>,
    pads: Option<Span<usize>>,
    strides: Option<Span<usize>>,
) -> Tensor<T> {
    let dilations = match dilations {
        Option::Some(dilations) => dilations,
        Option::None => {
            let mut dilations: Array<usize> = array![];
            let mut i = 0;
            while i != image_shape.len() {
                dilations.append(1);
                i += 1;
            };

            dilations.span()
        },
    };

    let pads = match pads {
        Option::Some(pads) => pads,
        Option::None => {
            let mut pads: Array<usize> = array![];
            let mut i = 0;
            while i != image_shape.len() {
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
            let mut strides: Array<usize> = array![];
            let mut i = 0;
            while i != image_shape.len() {
                strides.append(1);
                i += 1;
            };

            strides.span()
        },
    };

    let bl = prod(block_shape, 0);
    let C = *(*data).shape.at(1) / bl;

    let mut new_shape: Array<i32> = array![
        (*(*data).shape.at(0)).try_into().unwrap(), C.try_into().unwrap(), bl.try_into().unwrap()
    ];
    let mut i = 2;
    while i != (*data)
        .shape
        .len() {
            new_shape.append((*(*data).shape.at(i)).try_into().unwrap());
            i += 1;
        };

    let data = data.reshape(new_shape.span(), false);

    let mut res: Array<T> = array![];
    let data_stride = stride(data.shape);

    let mut n = 0;
    while n != *data
        .shape
        .at(0) {
            let mut c = 0;
            while c != *data
                .shape
                .at(1) {
                    let data_n_c = TensorTrait::new(
                        SpanTrait::slice(data.shape, 2, data.shape.len() - 2),
                        SpanTrait::slice(
                            data.data,
                            n * *data_stride.at(0) + c * *data_stride.at(1),
                            *data_stride.at(1)
                        )
                    );
                    let mut out = col2im_naive_implementation(
                        @data_n_c, image_shape, block_shape, dilations, pads, strides
                    );
                    let mut i = 0;
                    while i != out.len() {
                        res.append(out.at(i));
                        i += 1;
                    };

                    c += 1;
                };

            n += 1;
        };

    let mut new_shape = array![*data.shape.at(0), *data.shape.at(1)];
    let mut i = 0;
    while i != image_shape.len() {
        new_shape.append(*image_shape.at(i));
        i += 1;
    };

    TensorTrait::new(new_shape.span(), res.span())
}

fn get_image<T, +Drop<T>, +Copy<T>>(self: @Tensor<T>, row: usize) -> Span<T> {
    assert((*self).shape.len() == 2, 'Expected a 2D tensor');

    let row_length = *self.shape[1];
    let start = row * row_length;

    (*self).data.slice(start, row_length)
}

fn col2im_naive_implementation<
    T, MAG, +TensorTrait<T>, +NumberTrait<T, MAG>, +Copy<T>, +Drop<T>, +Add<T>,
>(
    data: @Tensor<T>,
    image_shape: Span<usize>,
    kernel_shape: Span<usize>,
    dilations: Span<usize>,
    pads: Span<usize>,
    strides: Span<usize>,
) -> NullableVec<T> {
    let n_dims = pads.len() / 2;

    col2im_shape_check(data, image_shape, kernel_shape, dilations, pads, strides);

    let mut dim_col: Array<usize> = array![];
    let mut i = 0;
    while i != n_dims {
        dim_col
            .append(
                (*image_shape.at(i)
                    + (*pads.at(i) + *pads.at(i + n_dims))
                    - (*dilations.at(i) * (*kernel_shape.at(i) - 1) + 1))
                    / *strides.at(i)
                    + 1
            );

        i += 1;
    };

    let dim_col = dim_col.span();

    let stride_img = stride(image_shape);

    let mut data_im = NullableVecImpl::new();
    data_im.set(*image_shape.at(0) * *stride_img.at(0) - 1, NumberTrait::zero());

    let kernel_size = prod(kernel_shape, 0);
    let col_size = prod(dim_col, 0);
    let mut c_col = 0;
    while c_col != kernel_size {
        let offset = get_indices(c_col, kernel_shape).span();

        let mut col = 0;
        while col != col_size {
            let ind_col = get_indices(col, dim_col).span();
            let mut ind_im: Array<usize> = array![];
            let mut i = 0;
            while i != n_dims {
                if (*ind_col.at(i) * *strides.at(i) + *offset.at(i) * *dilations.at(i)) < *pads
                    .at(i) {
                    let neg_index = *pads.at(i)
                        - (*ind_col.at(i) * *strides.at(i) + *offset.at(i) * *dilations.at(i));
                    ind_im.append(*image_shape.at(i) + neg_index);
                } else {
                    ind_im
                        .append(
                            *ind_col.at(i) * *strides.at(i)
                                + *offset.at(i) * *dilations.at(i)
                                - *pads.at(i)
                        );
                }

                i += 1;
            };

            let ind_im = ind_im.span();
            if !is_out(ind_im, image_shape) {
                let mut index = 0;
                let mut i = 0;
                while i != image_shape.len() {
                    index += *stride_img.at(i) * *ind_im.at(i);
                    i += 1;
                };

                data_im.set(index, data_im.at(index) + *(*data).data.at(c_col * col_size + col));
            }

            col += 1;
        };

        c_col += 1;
    };

    data_im
}

fn col2im_shape_check<T, +TensorTrait<T>, +Copy<T>, +Drop<T>,>(
    X: @Tensor<T>,
    output_shape: Span<usize>,
    kernel_shape: Span<usize>,
    dilations: Span<usize>,
    pads: Span<usize>,
    strides: Span<usize>,
) {
    let n_input_plane = *(*X).shape.at(0);

    let kernel_size = prod(kernel_shape, 0);

    assert(n_input_plane % kernel_size == 0, 'wrong input dimension');

    let input_length = *(*X).shape.at(1);
    let n_dims = output_shape.len();
    let mut n_blocks: Array<usize> = array![];

    let mut i = 0;
    while i != n_dims {
        n_blocks
            .append(
                (*output_shape.at(i)
                    + (*pads.at(i) + *pads.at(i + n_dims))
                    - *dilations.at(i) * (*kernel_shape.at(i) - 1)
                    - 1)
                    / *strides.at(i)
                    + 1
            );
        i += 1;
    };

    let block_size = prod(n_blocks.span(), 0);

    assert(input_length == block_size, 'input_length != block_size');
}

fn get_indices(index: usize, shape: Span<usize>,) -> Array<usize> {
    let mut i = index;
    let mut res: Array<usize> = array![];
    let mut k = shape.len() - 1;
    while k != 0 {
        let m = i % *shape.at(k);
        res.append(m);
        i -= m;
        i /= *shape.at(k);
        k -= 1;
    };

    let mut new_res: Array<usize> = array![];
    new_res.append(i);
    let mut i = shape.len() - 1;
    while i != 0 {
        new_res.append(*res.at(i - 1));
        i -= 1;
    };

    new_res
}

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

    is_out
}

fn prod<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TensorTrait<T>, +Mul<T>,>(
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
