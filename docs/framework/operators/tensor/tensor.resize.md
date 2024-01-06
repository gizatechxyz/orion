#tensor.resize

```rust
    fn resize(
        self: @Tensor<T>,
        roi: Option<Tensor<T>>,
        scales: Option<Span<T>>,
        sizes: Option<Span<usize>>,
        antialias: Option<usize>,
        axes: Option<Span<usize>>,
        coordinate_transformation_mode: Option<orion::operators::tensor::math::resize::TRANSFORMATION_MODE>,
        cubic_coeff_a: Option<T>,
        exclude_outside: Option<bool>,
        extrapolation_value: Option<T>,
        keep_aspect_ratio_policy: Option<orion::operators::tensor::math::resize::KEEP_ASPECT_RATIO_POLICY>,
        mode: Option<orion::operators::tensor::math::resize::MODE>,
        nearest_mode: Option<orion::operators::tensor::math::resize::NEAREST_MODE>,
    ) -> Tensor<T>;
```

Resizes the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood in the input tensor. 

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `roi` (`Option<Tensor<T>>`) (optional) - 1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X or the length of axes, if provided. It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
* `scales` (`Option<Tensor<T>>`) (optional) - The scale array along each dimension. It takes value greater than 0. If it's less than 1, it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should be the same as the rank of input 'X' or the length of 'axes', if provided. One and only one of 'scales' and 'sizes' MUST be specified.
* `sizes` (`Option<Tensor<usize>>`) (optional) - Target size of the output tensor. Its interpretation depends on the 'keep_aspect_ratio_policy' value. The number of elements of 'sizes' should be the same as the rank of input 'X', or the length of 'axes', if provided. One and only one of 'scales' and 'sizes' MUST be specified.  
* `antialias` (`Option<usize>`) (default is 0) - If set to 1, "linear" and "cubic" interpolation modes will use an antialiasing filter when downscaling. Antialiasing is achieved by stretching the resampling filter by a factor max(1, 1 / scale).
* `axes`(`Option<Span<usize>>`) - If provided, it specifies a subset of axes that 'roi', 'scales' and 'sizes' refer to. If not provided, all axes are assumed [0, 1, ..., r-1], where r = rank(data). 
* `coordinate_transformation_mode` (`Option<TRANSFORMATION_MODE>`) (default is half_pixel) - This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor. 
* `cubic_coeff_a` (`Option<T>`) (default is -0.75) - The coefficient 'a' used in cubic interpolation.
* `exclude_outside` (`Option<bool>`) (default is false) - If set to true, the weight of sampling locations outside the tensor will be set to 0 and the weight will be renormalized so that their sum is 1.0. 
* `extrapolation_value` (`Option<T>`) (default is 0.0) - When coordinate_transformation_mode is "tf_crop_and_resize" and x_original is outside the range [0, length_original - 1], this value is used as the corresponding output value. 
* `keep_aspect_ratio_policy` (`Option<KEEP_ASPECT_RATIO_POLICY>`) (default is stretch) - This attribute describes how to interpret the `sizes` input with regard to keeping the original aspect ratio of the input, and it is not applicable when the `scales` input is used. 
* `mode` (`Option<MODE>`) (default is nearest) - Three interpolation modes: "nearest", "linear" and "cubic".
* `nearest_mode` (`Option<NEAREST_MODE>`) (default is round_prefer_floor) - Four modes: "round_prefer_floor" (as known as round half down), "round_prefer_ceil" (as known as round half up), "floor", "ceil". Only used by nearest interpolation. 

## Panics

* Panics if both scales and sizes are `Option::None`.
* Panics if roi is `Option::None` for the coordinate_transformation_mode `tf_crop_and_resize`.
* Panics if antialias is not `Option::None` for mode `nearest`.

## Returns

A new resized `Tensor<T>` of the dimension given by output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) is scale is specified, or output_size if size is specified (note that some value of the parameter `keep_aspect_ratio_policy` can change sizes and therefore the dimension of the output tensor) 

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor, FP16x16TensorPartialEq};
use orion::operators::tensor::math::resize::{
    MODE, NEAREST_MODE, KEEP_ASPECT_RATIO_POLICY, TRANSFORMATION_MODE
};
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};
use core::debug::PrintTrait;

fn example_resize_downsample_scales_linear() -> Tensor<FP16x16>{
    let mut data = TensorTrait::<
        FP16x16
    >::new(
        shape: array![1, 1, 2, 4].span(),
        data: array![
            FixedTrait::<FP16x16>::new(65536, false),   //1
            FixedTrait::<FP16x16>::new(131072, false),  //2
            FixedTrait::<FP16x16>::new(196608, false),  //3
            FixedTrait::<FP16x16>::new(262144, false),  //4
            FixedTrait::<FP16x16>::new(327680, false),  //5
            FixedTrait::<FP16x16>::new(393216, false),  //6
            FixedTrait::<FP16x16>::new(458752, false),  //7
            FixedTrait::<FP16x16>::new(524288, false),  //8
        ]
            .span(),
    );
    let mut scales = array![
        FixedTrait::<FP16x16>::new(65536, false),  //1
        FixedTrait::<FP16x16>::new(65536, false),   
        FixedTrait::<FP16x16>::new(39322, false),  //0.6
        FixedTrait::<FP16x16>::new(39322, false)
    ]
        .span();

    let scales = Option::Some(scales);

    return data.resize(
        Option::None,
        scales,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::Some(MODE::LINEAR),
        Option::None,
    );

}
>>> [[[[2.6666665 4.3333331]]]]



fn example_resize_tf_crop_and_resize_extrapolation_value() -> Tensor<FP16x16> {
    let mut data = TensorTrait::<
        FP16x16
    >::new(
        shape: array![1, 1, 4, 4].span(),
        data: array![
            FixedTrait::<FP16x16>::new(65536, false),
            FixedTrait::<FP16x16>::new(131072, false),
            FixedTrait::<FP16x16>::new(196608, false),
            FixedTrait::<FP16x16>::new(262144, false),
            FixedTrait::<FP16x16>::new(327680, false),
            FixedTrait::<FP16x16>::new(393216, false),
            FixedTrait::<FP16x16>::new(458752, false),
            FixedTrait::<FP16x16>::new(524288, false),
            FixedTrait::<FP16x16>::new(589824, false),
            FixedTrait::<FP16x16>::new(655360, false),
            FixedTrait::<FP16x16>::new(720896, false),
            FixedTrait::<FP16x16>::new(786432, false),
            FixedTrait::<FP16x16>::new(851968, false),
            FixedTrait::<FP16x16>::new(917504, false),
            FixedTrait::<FP16x16>::new(983040, false),
            FixedTrait::<FP16x16>::new(1048576, false),
        ]
            .span(),
    );

    let mut roi = TensorTrait::<
        FP16x16
    >::new(
        shape: array![8].span(),
        data: array![
            FixedTrait::<FP16x16>::new(0, false),
            FixedTrait::<FP16x16>::new(0, false),
            FixedTrait::<FP16x16>::new(26214, false),
            FixedTrait::<FP16x16>::new(39322, false),
            FixedTrait::<FP16x16>::new(65536, false),
            FixedTrait::<FP16x16>::new(65536, false),
            FixedTrait::<FP16x16>::new(78643, false),
            FixedTrait::<FP16x16>::new(111411, false),
        ]
            .span(),
    );
    let roi = Option::Some(roi);

    let mut sizes = array![1, 1, 3, 3].span();
    let sizes = Option::Some(sizes);

    let extrapolation_value = Option::Some(FixedTrait::<FP16x16>::new(655360, false));

    return data.resize(
        roi,
        Option::None,
        sizes,
        Option::None,
        Option::None,
        Option::Some(TRANSFORMATION_MODE::TF_CROP_AND_RESIZE),
        Option::None,
        Option::None,
        extrapolation_value,
        Option::None,
        Option::Some(MODE::LINEAR),
        Option::None,
    );

}
>>> [[[[ 7.6000004 10.        10.       ]
    [12.400001  10.        10.       ]
    [10.        10.        10.       ]]]]



fn example_resize_downsample_sizes_cubic_antialias() -> Tensor<FP16x16> {
    let mut data = TensorTrait::<
        FP16x16
    >::new(
        shape: array![1, 1, 4, 4].span(),
        data: array![
            FixedTrait::<FP16x16>::new(65536, false),
            FixedTrait::<FP16x16>::new(131072, false),
            FixedTrait::<FP16x16>::new(196608, false),
            FixedTrait::<FP16x16>::new(262144, false),
            FixedTrait::<FP16x16>::new(327680, false),
            FixedTrait::<FP16x16>::new(393216, false),
            FixedTrait::<FP16x16>::new(458752, false),
            FixedTrait::<FP16x16>::new(524288, false),
            FixedTrait::<FP16x16>::new(589824, false),
            FixedTrait::<FP16x16>::new(655360, false),
            FixedTrait::<FP16x16>::new(720896, false),
            FixedTrait::<FP16x16>::new(786432, false),
            FixedTrait::<FP16x16>::new(851968, false),
            FixedTrait::<FP16x16>::new(917504, false),
            FixedTrait::<FP16x16>::new(983040, false),
            FixedTrait::<FP16x16>::new(1048576, false),
        ]
            .span(),
    );

    let antialias = Option::Some(1);

    let mut sizes = array![1, 1, 3, 3].span();
    let sizes = Option::Some(sizes);

    return data.resize(
        Option::None,
        Option::None,
        sizes,
        antialias,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::Some(MODE::CUBIC),
        Option::None,
    );
}

>>> [[[[ 1.7750092  3.1200073  4.4650054]
    [ 7.1550016  8.5        9.844998 ]
    [12.534994  13.8799925 15.224991 ]]]]

```
