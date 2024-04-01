# NNTrait::roi_align

```rust
    fn roi_align(
    X: @Tensor<T>,
    roi: @Tensor<T>,
    batch_indices: @Tensor<usize>,
    coordinate_transformation_mode: Option<
        orion::operators::nn::functional::roi_align::TRANSFORMATION_MODE
    >,
    mode: Option<orion::operators::nn::functional::roi_align::MODE>,
    output_height: Option<usize>,
    output_width: Option<usize>,
    sampling_ratio: Option<T>,
    spatial_scale: Option<T>,
) -> Tensor<T>;
```

 RoiAlign consumes an input tensor X and region of interests (rois) to apply pooling across each RoI; it produces a 4-D tensor of shape (num_rois, C, output_height, output_width).

## Args

* `X`(`@Tensor<T>`) - Input data tensor from the previous operator; 4-D feature map of shape (N, C, H, W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.
* `rois`(`@Tensor<T>`) - RoIs (Regions of Interest) to pool over; rois is 2-D input of shape (num_rois, 4) given as [[x1, y1, x2, y2], ...].
* `batch_indices`(`@Tensor<usize>`) - 1-D tensor of shape (num_rois,) with each element denoting the index of the corresponding image in the batch.
* `coordinate_transformation_mode`(`Option<TRANSFORMATION_MODE>`) - Allowed values are 'half_pixel' and 'output_half_pixel'. Use the value 'half_pixel' to pixel shift the input coordinates by -0.5 (default behavior). Use the value 'output_half_pixel' to omit the pixel shift for the input
* `mode`(`Option<MODE>`) -The pooling method. Two modes are supported: 'avg' and 'max'. Default is 'avg'.
* `output_height`(`Option<usize>`) - default 1; Pooled output Y's height.
* `output_width`(`Option<usize>`) - default 1; Pooled output Y's width.
* `sampling_ratio`(`Option<T>`) - Number of sampling points in the interpolation grid used to compute the output value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio grid points are used. If == 0, then an adaptive number of grid points are used (computed as ceil(roi_width / output_width), and likewise for height). Default is 0.
* `spatial_scale`(`Option<T>`) - Multiplicative spatial scale factor to translate ROI coordinates from their input spatial scale to the scale used when pooling, i.e., spatial scale of the input feature map X relative to the input image. Default is 1.0. 

## Returns

A `Tensor<T>` RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). The r-th batch element Y[r-1] is a pooled feature map corresponding to the r-th RoI X[r-1].

## Example

```rust
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::operators::nn::functional::roi_align::TRANSFORMATION_MODE;

fn example_roi_align() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(5);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 18114, sign: false });
    data.append(FP16x16 { mag: 46858, sign: false });
    data.append(FP16x16 { mag: 12831, sign: false });
    data.append(FP16x16 { mag: 22387, sign: false });
    data.append(FP16x16 { mag: 30395, sign: false });
    data.append(FP16x16 { mag: 63157, sign: false });
    data.append(FP16x16 { mag: 5865, sign: false });
    data.append(FP16x16 { mag: 19129, sign: false });
    data.append(FP16x16 { mag: 44256, sign: false });
    data.append(FP16x16 { mag: 1533, sign: false });
    data.append(FP16x16 { mag: 21397, sign: false });
    data.append(FP16x16 { mag: 55567, sign: false });
    data.append(FP16x16 { mag: 63556, sign: false });
    data.append(FP16x16 { mag: 16193, sign: false });
    data.append(FP16x16 { mag: 61184, sign: false });
    data.append(FP16x16 { mag: 1350, sign: false });
    data.append(FP16x16 { mag: 11272, sign: false });
    data.append(FP16x16 { mag: 14123, sign: false });
    data.append(FP16x16 { mag: 28796, sign: false });
    data.append(FP16x16 { mag: 4279, sign: false });
    data.append(FP16x16 { mag: 26620, sign: false });
    data.append(FP16x16 { mag: 14378, sign: false });
    data.append(FP16x16 { mag: 29314, sign: false });
    data.append(FP16x16 { mag: 30716, sign: false });
    data.append(FP16x16 { mag: 46589, sign: false });
    let mut X = TensorTrait::new(shape.span(), data.span());

    let batch_indices = TensorTrait::new(array![3].span(), array![0, 0, 0].span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(FP16x16 { mag: 589824, sign: false });
    let rois = TensorTrait::new(shape.span(), data.span());

    return roi_align(
        @X,
        @rois,
        @batch_indices,
        Option::Some(TRANSFORMATION_MODE::OUTPUT_HALF_PIXEL),
        Option::None,
        Option::Some(2),
        Option::Some(2),
        Option::Some(FP16x16 { mag: 65536, sign: false }),
        Option::Some(FP16x16 { mag: 32768, sign: false }),
    );
}
>>> [[[[0.2083422 , 0.44005   ],
        [0.20385626, 0.39676717]]],


      [[[0.09630001, 0.19375   ],
        [0.3128    , 0.33335   ]]],


      [[[0.4394    , 0.0653    ],
        [0.4687    , 0.7109    ]]]]

````