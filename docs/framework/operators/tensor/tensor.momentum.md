# tensor.momentum

```rust
fn momentum(
    r: T, t: T, inputs: @Tensor<T>, alpha: T, beta: T, mode: MODE, norm_coefficient: T,
) -> (Tensor<T>, Tensor<T>);
```

Compute one iteration of stochastic gradient update with momentum.

## Args

* `r`(`T`) - The learning rate.
* `i`(`T`) - Update count of "X".
* `inputs`(`@Tensor<T>`) - It sequentially contains the current values of optimized tensors, then their gradient tensors, and finally their momentum tensors. For example, if two tensors "X_1" and "X_2" are optimized, The expected input list would be ["X_1", "X_2", gradient of "X_1", gradient of "X_2", momentum of "X_1", momentum of "X_2"].
* `alpha`(`T`) - The decay factor of momentum.
* `beta`(`T`) - The coefficient of gradient in computing new momentum. 
* `mode`(`MODE`) - Its value should be either "nesterov" or "standard". The value "nesterov" leads to the use of Nesterov's momentum while "standard" invokes stochastic gradient method using standard momentum
* `norm_coefficient`(`T`) - Coefficient of 0.5 * norm_coefficient * ||X||^2.
## Returns

Two `Tensor<T>` containing the new values of optimized tensors and then the new values of their momentum tensors.

## Type Constraints

* `T` in (`Tensor<FP>`, `Tensor<i8>`, `Tensor<i32>`, `tensor<u32>,`)

## Examples

```rust
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{FP16x16Tensor};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::preview_training::momentum::MODE;

fn example_momentum() -> (Tensor<FP16x16>, Tensor<FP16x16>){
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 78643, sign: false });
    data.append(FP16x16 { mag: 183500, sign: false });
    data.append(FP16x16 { mag: 61603, sign: true });
    data.append(FP16x16 { mag: 163840, sign: true });
    data.append(FP16x16 { mag: 111411, sign: false });
    data.append(FP16x16 { mag: 235929, sign: false });
    let mut X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65, sign: false });
    data.append(FP16x16 { mag: 62259, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    let param = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 74211, sign: false });
    data.append(FP16x16 { mag: 177453, sign: false });
    let expected_output = TensorTrait::new(shape.span(), data.span());


    return TensorTrait::momentum(
        FP16x16 { mag: 6553, sign: false },
        FP16x16 { mag: 0, sign: false },
        @X,
        *param.data.at(1),
        *param.data.at(2),
        MODE::STANDARD,
        *param.data.at(0),
    );
}
>>> ([1.13238 2.70772],[0.67620003 0.9227998 ])

```
