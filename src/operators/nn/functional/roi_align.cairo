use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

#[derive(Copy, Drop, Destruct)]
struct PreCalc<T> {
    pos1: usize,
    pos2: usize,
    pos3: usize,
    pos4: usize,
    w1: T,
    w2: T,
    w3: T,
    w4: T,
}

#[derive(Copy, Drop)]
enum MODE {
    AVG,
    MAX,
}

#[derive(Copy, Drop)]
enum TRANSFORMATION_MODE {
    HALF_PIXEL,
    OUTPUT_HALF_PIXEL,
}

/// Cf: TensorTrait::roi_align docstring
fn roi_align<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
    +Neg<T>,
    +DivEq<T>,
>(
    X: @Tensor<T>,
    roi: @Tensor<T>,
    batch_indices: @Tensor<usize>,
    coordinate_transformation_mode: Option<TRANSFORMATION_MODE>,
    mode: Option<MODE>,
    output_height: Option<usize>,
    output_width: Option<usize>,
    sampling_ratio: Option<T>,
    spatial_scale: Option<T>,
) -> Tensor<T> {
    let num_channels = *(*X).shape.at(1);
    let num_rois = *(*batch_indices).shape.at(0);
    let num_roi_cols = *(*roi).shape.at(1);

    let output_height = match output_height {
        Option::Some(output_height) => output_height,
        Option::None => 1,
    };

    let output_width = match output_width {
        Option::Some(output_width) => output_width,
        Option::None => 1,
    };

    let coordinate_transformation_mode = match coordinate_transformation_mode {
        Option::Some(coordinate_transformation_mode) => coordinate_transformation_mode,
        Option::None => TRANSFORMATION_MODE::HALF_PIXEL,
    };

    let half_pixel = match coordinate_transformation_mode {
        TRANSFORMATION_MODE::HALF_PIXEL => true,
        TRANSFORMATION_MODE::OUTPUT_HALF_PIXEL => false,
    };

    let y_dims = array![num_rois, num_channels, output_height, output_width].span();

    let y_data = roi_align_forward(
        y_dims,
        (*X).data,
        spatial_scale,
        *(*X).shape.at(2),
        *(*X).shape.at(3),
        sampling_ratio,
        (*roi).data,
        num_roi_cols,
        mode,
        half_pixel,
        (*batch_indices).data
    );

    return TensorTrait::new(y_dims, y_data);
}


fn roi_align_forward<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
    +Neg<T>,
    +DivEq<T>,
>(
    output_shape: Span<usize>,
    bottom_data: Span<T>,
    spatial_scale: Option<T>,
    height: usize,
    width: usize,
    sampling_ratio: Option<T>,
    bottom_rois: Span<T>,
    num_roi_cols: usize,
    mode: Option<MODE>,
    half_pixel: bool,
    batch_indices_ptr: Span<usize>,
) -> Span<T> {
    let n_rois = *output_shape.at(0);
    let channels = *output_shape.at(1);
    let pooled_height = *output_shape.at(2);
    let pooled_width = *output_shape.at(3);

    let mut top_data = ArrayTrait::new();

    let spatial_scale = match spatial_scale {
        Option::Some(spatial_scale) => spatial_scale,
        Option::None => NumberTrait::one(),
    };

    let sampling_ratio = match sampling_ratio {
        Option::Some(sampling_ratio) => sampling_ratio,
        Option::None => NumberTrait::zero(),
    };

    let mode = match mode {
        Option::Some(mode) => mode,
        Option::None => MODE::AVG,
    };

    let mut n = 0;

    while n != n_rois {
        let offset_bottom_rois = n * num_roi_cols;
        let roi_batch_ind = *batch_indices_ptr.at(n);

        let offset: T = if half_pixel {
            NumberTrait::half()
        } else {
            NumberTrait::zero()
        };

        let roi_start_w = *bottom_rois.at(offset_bottom_rois + 0) * spatial_scale - offset;
        let roi_start_h = *bottom_rois.at(offset_bottom_rois + 1) * spatial_scale - offset;
        let roi_end_w = *bottom_rois.at(offset_bottom_rois + 2) * spatial_scale - offset;
        let roi_end_h = *bottom_rois.at(offset_bottom_rois + 3) * spatial_scale - offset;

        let mut roi_width = roi_end_w - roi_start_w;
        let mut roi_height = roi_end_h - roi_start_h;

        if !half_pixel {
            roi_width = NumberTrait::max(roi_width, NumberTrait::one());
            roi_height = NumberTrait::max(roi_height, NumberTrait::one());
        }

        let bin_size_h = roi_height / NumberTrait::new_unscaled(pooled_height.into(), false);
        let bin_size_w = roi_width / NumberTrait::new_unscaled(pooled_width.into(), false);

        let roi_bin_grid_h: usize = if sampling_ratio > NumberTrait::zero() {
            sampling_ratio.try_into().unwrap()
        } else {
            NumberTrait::<
                T
            >::ceil(roi_height / NumberTrait::new_unscaled(pooled_height.into(), false))
                .try_into()
                .unwrap()
        };

        let roi_bin_grid_w: usize = if sampling_ratio > NumberTrait::zero() {
            sampling_ratio.try_into().unwrap()
        } else {
            NumberTrait::<
                T
            >::ceil(roi_width / NumberTrait::new_unscaled(pooled_width.into(), false))
                .try_into()
                .unwrap()
        };

        let count = NumberTrait::max(roi_bin_grid_h * roi_bin_grid_w, 1);
        let mut pre_calc = ArrayTrait::new();
        let pre_calc = pre_calc_for_bilinear_interpolate(
            height,
            width,
            pooled_height,
            pooled_width,
            roi_bin_grid_h,
            roi_bin_grid_w,
            roi_start_h,
            roi_start_w,
            bin_size_h,
            bin_size_w,
            roi_bin_grid_h,
            roi_bin_grid_w,
            ref pre_calc,
        );

        let mut c = 0;
        while c != channels {
            let offset_bottom_data = (roi_batch_ind * channels + c) * height * width;

            let mut pre_calc_index = 0;
            let mut ph = 0;
            while ph != pooled_height {
                let mut pw = 0;
                while pw != pooled_width {
                    let mut output_val = NumberTrait::zero();

                    match mode {
                        MODE::AVG => {
                            let mut _iy = 0;
                            while _iy != roi_bin_grid_h {
                                let mut _ix = 0;
                                while _ix != roi_bin_grid_w {
                                    let pc = *pre_calc.at(pre_calc_index);

                                    output_val +=
                                        (pc.w1 * *bottom_data.at(offset_bottom_data + pc.pos1)
                                            + pc.w2 * *bottom_data.at(offset_bottom_data + pc.pos2)
                                            + pc.w3 * *bottom_data.at(offset_bottom_data + pc.pos3)
                                            + pc.w4
                                                * *bottom_data.at(offset_bottom_data + pc.pos4));
                                    pre_calc_index += 1;
                                    _ix += 1;
                                };

                                _iy += 1;
                            };

                            output_val /= NumberTrait::new_unscaled(count.into(), false);
                            top_data.append(output_val);
                        },
                        MODE::MAX => {
                            let mut max_flag = false;
                            let mut _iy = 0;
                            while _iy != roi_bin_grid_h {
                                let mut _ix = 0;
                                let mut val = NumberTrait::zero();

                                while _ix != roi_bin_grid_w {
                                    let pc = *pre_calc.at(pre_calc_index);

                                    val =
                                        NumberTrait::max(
                                            NumberTrait::max(
                                                pc.w1
                                                    * *bottom_data.at(offset_bottom_data + pc.pos1),
                                                pc.w2
                                                    * *bottom_data.at(offset_bottom_data + pc.pos2)
                                            ),
                                            NumberTrait::max(
                                                pc.w3
                                                    * *bottom_data.at(offset_bottom_data + pc.pos3),
                                                pc.w4
                                                    * *bottom_data.at(offset_bottom_data + pc.pos4)
                                            )
                                        );
                                    if !max_flag {
                                        output_val = val;
                                        max_flag = true;
                                    } else {
                                        output_val = NumberTrait::max(output_val, val);
                                    }
                                    pre_calc_index += 1;

                                    _ix += 1;
                                };
                                _iy += 1;
                            };
                            top_data.append(output_val);
                        }
                    }
                    pw += 1;
                };
                ph += 1;
            };

            c += 1;
        };
        n += 1;
    };

    return top_data.span();
}


fn pre_calc_for_bilinear_interpolate<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
    +Neg<T>,
>(
    height: usize,
    width: usize,
    pooled_height: usize,
    pooled_width: usize,
    iy_upper: usize,
    ix_upper: usize,
    roi_start_h: T,
    roi_start_w: T,
    bin_size_h: T,
    bin_size_w: T,
    roi_bin_grid_h: usize,
    roi_bin_grid_w: usize,
    ref pre_calc: Array<PreCalc<T>>
) -> Span<PreCalc<T>> {
    let mut pre_calc_index = 0;

    let roi_bin_grid_h = NumberTrait::new_unscaled(roi_bin_grid_h.into(), false);
    let roi_bin_grid_w = NumberTrait::new_unscaled(roi_bin_grid_w.into(), false);

    let height = NumberTrait::new_unscaled(height.into(), false);
    let width = NumberTrait::new_unscaled(width.into(), false);

    let mut ph = 0;
    while ph != pooled_height {
        let mut pw = 0;
        while pw != pooled_width {
            let mut iy: usize = 0;
            while iy != iy_upper {
                let yy = roi_start_h
                    + NumberTrait::new_unscaled(ph.into(), false) * bin_size_h
                    + (NumberTrait::new_unscaled(iy.into(), false) + NumberTrait::half())
                        * bin_size_h
                        / roi_bin_grid_h;

                let mut ix = 0;
                while ix != ix_upper {
                    let xx = roi_start_w
                        + NumberTrait::new_unscaled(pw.into(), false) * bin_size_w
                        + (NumberTrait::new_unscaled(ix.into(), false) + NumberTrait::half())
                            * bin_size_w
                            / roi_bin_grid_w;
                    let mut x: T = xx;
                    let mut y: T = yy;

                    if y < -NumberTrait::one()
                        || y > height
                        || x < -NumberTrait::one()
                        || x > width {
                        let pc: PreCalc<T> = PreCalc {
                            pos1: 0,
                            pos2: 0,
                            pos3: 0,
                            pos4: 0,
                            w1: NumberTrait::zero(),
                            w2: NumberTrait::zero(),
                            w3: NumberTrait::zero(),
                            w4: NumberTrait::zero()
                        };
                        pre_calc.append(pc);
                        pre_calc_index += 1;
                    } else {
                        y = NumberTrait::max(y, NumberTrait::zero());
                        x = NumberTrait::max(x, NumberTrait::zero());

                        let mut y_low = NumberTrait::floor(y);
                        let mut x_low = NumberTrait::floor(x);

                        let mut y_high = y_low;
                        let mut x_high = x_low;

                        if y_low >= height - NumberTrait::one() {
                            y_low = height - NumberTrait::one();
                            y_high = y_low;
                            y = y_low;
                        } else {
                            y_high += NumberTrait::one();
                        };

                        if x_low >= width - NumberTrait::one() {
                            x_low = width - NumberTrait::one();
                            x_high = x_low;
                            x = x_low;
                        } else {
                            x_high += NumberTrait::one();
                        };

                        let ly = y - y_low;
                        let lx = x - x_low;
                        let hy = NumberTrait::one() - ly;
                        let hx = NumberTrait::one() - lx;
                        let w1 = hy * hx;
                        let w2 = hy * lx;
                        let w3 = ly * hx;
                        let w4 = ly * lx;

                        let pc: PreCalc<T> = PreCalc {
                            pos1: (y_low * width + x_low).try_into().unwrap(),
                            pos2: (y_low * width + x_high).try_into().unwrap(),
                            pos3: (y_high * width + x_low).try_into().unwrap(),
                            pos4: (y_high * width + x_high).try_into().unwrap(),
                            w1: w1,
                            w2: w2,
                            w3: w3,
                            w4: w4
                        };
                        pre_calc.append(pc);

                        pre_calc_index += 1;
                    }

                    ix += 1;
                };

                iy += 1;
            };
            pw += 1;
        };
        ph += 1;
    };

    return pre_calc.span();
}

