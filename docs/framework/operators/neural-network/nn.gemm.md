# NNTrait::gemm

```rust
    fn gemm(
        A: Tensor<T>,
        B: Tensor<T>,
        C: Option<Tensor<T>>,
        alpha: Option<T>,
        beta: Option<T>,
        transA: bool,
        transB: bool
    ) -> Tensor<T>;
```

Performs General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

* A' = transpose(A) if transA else A
* B' = transpose(B) if transB else B

Compute `Y = alpha * A' * B' + beta * C`, where input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N).
`A` will be transposed before doing the computation if attribute `transA` is `true`, same for `B` and `transB`.

## Args

* `A`(`Tensor<T>`) - Input tensor A. The shape of `A` should be (M, K) if `transA` is `false`, or (K, M) if `transA` is `true`.
* `B`(`Tensor<T>`) - Input tensor B. The shape of `B` should be (K, N) if `transB` is `false`, or (N, K) if `transB` is `true`.
* `C`(`Option<Tensor<T>>`) - Optional input tensor C. The shape of C should be unidirectional broadcastable to (M, N). 
* `alpha`(`Option<T>`) - Optional scalar multiplier for the product of input tensors `A * B`.
* `beta`(`Option<T>`) - Optional scalar multiplier for input tensor `C`.
* `transA`(`bool`) - Whether `A` should be transposed.
* `transB`(`bool`) - Whether `B` should be transposed.

## Returns

A `Tensor<T>` of shape (M, N).

## Examples

```rust
    mod input_0;
    mod input_1;
    mod input_2;
    
    use orion::operators::nn::NNTrait;
    use orion::numbers::FixedTrait;
    use orion::operators::nn::FP16x16NN;
    use orion::operators::tensor::FP16x16TensorPartialEq;

  fn gemm_all_attributes_example() -> Tensor<FP16x16> {
      let input_0 = input_0::input_0(); // shape [4;3]
      let input_1 = input_1::input_1(); // shape [5;4]
      let input_2 = input_2::input_2(); // shape [1;5]

      let y = NNTrait::gemm(
          input_0,
          input_1,
          Option::Some(input_2),
          Option::Some(FixedTrait::new(16384, false)), // 0.25
          Option::Some(FixedTrait::new(22938, false)), // 0.35
          true,
          true
      );

      return y;
  } 
 >>> tensor of shape [3;5]
````
