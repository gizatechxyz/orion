# How to start contributing

- To start contributing first you need to decide which ONNX Operator you want to implement in the Cairo Runtime. You can check the compatibility list here. Please search the [issue tracker](https://github.com/franalgaba/onnx-cairo/issues) for a similar idea first: there may already be an issue you can contribute to.

- Now you can start working in your new operator! First let's explore the folder structure for the project.
The project is organized into several modules, grouped by functionality. Each module has its own folder, which contains the main module file (e.g., matmul.cairo) and separate implementation files for different data types (e.g., matmul_i32.cairo, matmul_u32.cairo).

```
src
├── lib.cairo
├── operators
│   ├── linalg
│   │   ├── matmul
│   │   │   ├── helpers.cairo
│   │   │   ├── matmul_i32.cairo
│   │   │   └── matmul_u32.cairo
│   │   └── matmul.cairo
│   ├── linalg.cairo
│   ├── math
│   │   ├── argmax
│   │   │   ├── argmax_i32.cairo
│   │   │   └── argmax_u32.cairo
│   │   ├── argmax.cairo
│   │   ├── max
│   │   │   ├── max_i32.cairo
│   │   │   └── max_u32.cairo
│   │   ├── max.cairo
│   │   ├── min
│   │   │   ├── min_i32.cairo
│   │   │   └── min_u32.cairo
│   │   ├── min.cairo
│   │   ├── reduce_sum
│   │   │   ├── reduce_sum_i32.cairo
│   │   │   └── reduce_sum_u32.cairo
│   │   ├── reduce_sum.cairo
│   │   ├── signed_integer
│   │   │   ├── i128.cairo
│   │   │   ├── i16.cairo
│   │   │   ├── i32.cairo
│   │   │   ├── i64.cairo
│   │   │   ├── i8.cairo
│   │   │   └── integer_trait.cairo
│   │   └── signed_integer.cairo
│   ├── math.cairo
│   ├── nn
│   │   ├── functional
│   │   │   ├── relu
│   │   │   │   ├── relu_i32.cairo
│   │   │   │   └── relu_u32.cairo
│   │   │   └── relu.cairo
│   │   ├── functional.cairo
│   │   ├── nn_i32.cairo
│   │   └── nn_u32.cairo
│   ├── nn.cairo
│   ├── tensor
│   │   ├── core.cairo
│   │   ├── helpers.cairo
│   │   ├── tensor_i32.cairo
│   │   └── tensor_u32.cairo
│   └── tensor.cairo
├── operators.cairo
├── performance
│   ├── functional
│   │   ├── quantization
│   │   │   ├── quant_i32.cairo
│   │   │   └── quant_u32.cairo
│   │   └── quantization.cairo
│   ├── functional.cairo
│   ├── performance_i32.cairo
│   └── performance_u32.cairo
├── performance.cairo
├── tests
│   ├── operators
│   │   ├── linalg
│   │   │   └── matmul_test.cairo
│   │   ├── linalg.cairo
│   │   ├── math
│   │   │   ├── argmax_test.cairo
│   │   │   ├── max_test.cairo
│   │   │   ├── min_test.cairo
│   │   │   ├── reduce_sum_test.cairo
│   │   │   └── signed_integer_test.cairo
│   │   ├── math.cairo
│   │   ├── nn
│   │   │   └── relu_test.cairo
│   │   ├── nn.cairo
│   │   ├── tensor
│   │   │   ├── helpers.cairo
│   │   │   └── tensor_test.cairo
│   │   └── tensor.cairo
│   ├── operators.cairo
│   ├── performance
│   │   └── quantization_test.cairo
│   └── performance.cairo
├── tests.cairo
└── utils.cairo
```

## Creating New Methods

To create a new method or function in the library, follow these steps:

1. **Identify the appropriate module**: Determine which module your new method belongs to based on its functionality (e.g., linear algebra, math, neural networks, or tensors).

2. **Create the implementation files**: Create separate implementation files for different data types (e.g., `method_i32.cairo`, `method_u32.cairo`) within the corresponding module folder. If your method requires additional helper functions, create a `helpers.cairo` file in the same folder.

3. **Update the module's main file**: In the main module file (e.g., `matmul.cairo`), add the import statements for your new implementation files (e.g., `mod method_i32;`, `mod method_u32;`, `mod helpers;`).

4. **Implement the method**: Write the code for your new method in the appropriate implementation files, adhering to the project's coding conventions and structure.

5. **Write tests**: Create test cases for your new method in the `tests` folder, ensuring that your tests cover all relevant functionality and edge cases.

6. **Document your method**: Add comments and documentation in the code to explain the purpose, functionality, and usage of your new method. Update the README and other documentation files as necessary.

7. **Contribute your changes**: Follow the contribution steps outlined earlier in this guide to submit your new method to the main project repository.

## Add New Method to TensorTrait

1. Define the method to TensorTrait (`src/operators/tensor/core`) 

2. Add its method in each of the Tensor implementations (i32Tensor, u32Tensor...).
