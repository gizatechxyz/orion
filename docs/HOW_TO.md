# How to start contributing

* To start contributing first you need to decide which ONNX Operator you want to implement in the Cairo Runtime. You can check the compatibility list here. Please search the [issue tracker](https://github.com/franalgaba/onnx-cairo/issues) for a similar idea first: there may already be an issue you can contribute to.

* Now you can start working in your new operator! First let's explore the folder structure for the project:

```
src
├── lib.cairo
├── operators
│   ├── activations
│   │   ├── relu.cairo
│   │   └── softmax.cairo
│   ├── activations.cairo
│   ├── math
│   │   ├── int33.cairo
│   │   ├── matrix.cairo
│   │   ├── tensor.cairo
│   │   └── vector.cairo
│   └── math.cairo
├── operators.cairo
├── performance
│   └── quantizations.cairo
├── performance.cairo
├── tests
│   ├── int33_test.cairo
│   ├── matrix_test.cairo
│   ├── quantization_test.cairo
│   ├── relu_test.cairo
│   ├── softmax_test.cairo
│   └── vector_test.cairo
└── tests.cairo
```

* To add a new operation to the Matrix / Tensor implementation you have to implement it inside the `src/operators/math/matrix.cairo` following the same `impl` structure.

* To add a new activation implementation create a new file inside the `src/operators/activations` folder and follow the reference implementation for ReLU or Softmax. After you add your new activation add your new module definition inside the `src/operators/activations.cairo`.