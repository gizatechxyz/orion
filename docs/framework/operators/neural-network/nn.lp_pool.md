# NNTrait::lp_pool

```rust
    fn lp_pool(
        X: @Tensor<T>,
        auto_pad: Option<AUTO_PAD>,
        ceil_mode: Option<usize>,
        dilations: Option<Span<usize>>,
        kernel_shape: Span<usize>,
        p: Option<usize>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
        count_include_pad: Option<usize>,
    ) -> Tensor<T>;
```

LpPool consumes an input tensor X and applies Lp pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. Lp pooling consisting of computing the Lp norm on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

## Args

* `X`(`@Tensor<T>`) - Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. 
* `auto_pad`(`Option<AUTO_PAD>`) - Default is NOTSET, auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. NOTSET means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`.
* `ceil_mode`(`Option<usize>`) - Default is 0, Whether to use ceil or floor (default) to compute the output shape.
* `dilations`(`Option<Span<usize>>`) - Dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.
* `kernel_shape`(`Span<usize>`) - The size of the kernel along each axis.
* `p`(`Option<usize>`) - Default is 2, p value of the Lp norm used to pool over the input data.
* `pads`(`Option<Span<usize>>`) - Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.
* `strides`(`Option<Span<usize>>`) - Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
* `count_include_pad`(`Option<usize>`) - Default is 0, Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad.

## Returns

A `Tensor<T>` - output data tensor from Lp pooling across the input tensor.

## Examples
    
```rust
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};


fn lp_pool_example() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(4);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 393216, sign: false });
    data.append(FP16x16 { mag: 458752, sign: false });
    data.append(FP16x16 { mag: 524288, sign: false });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 720896, sign: false });
    data.append(FP16x16 { mag: 786432, sign: false });
    data.append(FP16x16 { mag: 851968, sign: false });
    data.append(FP16x16 { mag: 917504, sign: false });
    data.append(FP16x16 { mag: 983040, sign: false });
    data.append(FP16x16 { mag: 1048576, sign: false });
    let mut X = TensorTrait::new(shape.span(), data.span());

    return NNTrait::lp_pool(
        @X,
        Option::None,
        Option::None,
        Option::None,
        array![2, 2].span(),
        Option::Some(2),
        Option::None,
        Option::Some(array![1, 1].span()),
        Option::None,
    );
}


>>> [[[[ 8.124039,  9.899495, 11.74734 ],
       [15.556349, 17.492855, 19.442223],
       [23.366642, 25.337719, 27.313   ]]]]
 

````