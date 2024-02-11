# NNTrait::grid_sample

```rust
    fn grid_sample(
    X: @Tensor<T>,
    grid: @Tensor<T>,
    align_corner: Option<usize>,
    mode: Option<MODE>,
    padding_mode: Option<PADDING_MODE>,
) -> Tensor<T>;
```

Given an input X and a flow-field grid, computes the output Y using X values and pixel locations from the grid.

## Args

* `X`(`@Tensor<T>`) - Input tensor of shape (N, C, D1, D2, ..., Dr), where N is the batch size, C is the number of channels, D1, D2, ..., Dr are the spatial dimensions.
* `grid`(`@Tensor<T>`) - Input offset of shape (N, D1_out, D2_out, ..., Dr_out, r), where D1_out, D2_out, ..., Dr_out are the spatial dimensions of the grid and output, and r is the number of spatial dimensions. Grid specifies the sampling locations normalized by the input spatial dimensions. 
* `align_corners`(`Option<usize>`) - default is 0. If align_corners=1, the extrema are considered as referring to the center points of the input's corner pixels. If align_corners=0, they are instead considered as referring to the corner points of the input's corner pixels 
* `mode`(`Option<MODE>`) - default is linear. Three interpolation modes: linear (default), nearest and cubic.
* `padding_mode`(`Option<PADDING_MODE>`) - default is zeros. Support padding modes for outside grid values: `zeros`(default), `border`, `reflection`.

## Returns

A `Tensor<T>` of shape (N, C, D1_out, D2_out, ..., Dr_out) of the sampled values.

## Example

```rust
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};

fn example_grid_sample() -> Tensor<FP16x16> {

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 655360, sign: true });
    data.append(FP16x16 { mag: 655360, sign: true });
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });

    let mut grid = TensorTrait::new(shape.span(), data.span());


    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    let mut X = TensorTrait::new(shape.span(), data.span());


    return NNTrait::grid_sample(
        @X, @grid, Option::None, Option::None, Option::None,
    );

}

}
>>> [
        [
            [
                [0.0000, 0.0000, 1.7000, 0.0000], 
                [0.0000, 1.7000, 0.0000, 0.0000]
            ]
        ]
    ]

````