# Orion2 - A scalable and modular ZKML framework

Orion2 is a ZKML frameworks that uses composable ZK compilers to achieve modularity and scalability.

With Orion2 you only need to add 11 primitive operations (primops) to support a new ZK backend:
- **Unary** - Log2, Exp2, Sin, Sqrt, Recip
- **Binary** - Add, Mul, Mod, LessThan
- **Other** - SumReduce, MaxReduce, Contiguous

Every complex operation boils down to these primitive operations, so when you do `a - b` for instance, `add(a, mul(b, -1))` gets written to the graph. Or when you do `a.matmul(b)`, what actually gets put on the graph is `sum_reduce(mul(reshape(a), reshape(b)))`.

Once the graph is built, iterative ZK compiler passes can modify it to replace primops with more efficient ops, depending on the ZK backend it's running on. On CairoVM, for example, subtraction is native to Cairo, so the Orion Cairo compiler will look up `add(a, mul(b, -1))` sequence and replace it with the high-level operator `sub(a,b)`.

This approach leads to a simple library, and performance is only limited by the creativity of the compiler programmer, not the model programmer.

![Orion2](https://github.com/gizatechxyz/orion2/assets/113879115/75320a2f-336f-49c8-a40e-7538aee50f9d)
