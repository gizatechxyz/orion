# **Verifiable Lagrange interpolation**

Lagrange interpolation is a mathematical technique used to approximate a function that passes through a given set of points. It takes an input set of data points and computes a polynomial that passes through all of them. 

Given a set of $n+1$ data points (or interpolation nodes) $X_0, X_1, ..., X_n$ with corresponding function values $Y_0, Y_1, ..., Y_n$, Lagrange interpolation seeks to find a polynomial of degree at most $n$ that passes through all these points.

Below, we provide a brief review of the implementation of a Lagrange interpolation in Python, which we will then convert to Cairo to transform it into a verifiable ZKML (Lagrange interpolation), using the Orion library. 

Content overview:

1. [Lagrange interpolation with Python:](LagrangeInterpolationTutorial.md#used-dataset) We start with the basic implementation of Lagrange interpolation using Python.
2. [Convert your model to Cairo:](LagrangeInterpolationTutorial.md#convert-your-model-to-cairo) In the subsequent stage, we will create a new scarb project and replicate our model to Cairo which is a language for creating STARK-provable programs.
3. [Implementing Lagrange interpolation using Orion:](LagrangeInterpolationTutorial.md#implementing-lagrange-using-orion) To catalyze our development process, we will use the Orion Framework to construct the key functions to build our verifiable Lagrange interpolation.

### Used DataSet

In this tutorial, we will interpolate the Runge function, that we will define $f$ and uses the Chebyshev node as interpolation nodes.

```python
import numpy as np
import math
import matplotlib.pyplot as plt

# Runge function
def f(x):
    return 1 / (x**2 + 1)

# Chebyshev nodes
X = np.array( [5*math.cos(k*math.pi/10) for k in range(0,11)] )

Y = f(X)

```


### Implementation of Lagrange interpolating polynomial

Now we will implement Lagrange interpolation function in python.

Given a set of $n+1$ data points $X_0, X_1, ..., X_n$ with corresponding function values $Y_0, Y_1, ..., Y_n$, the Lagrange interpolating polynomial is

$$
    L(x) = \sum_{i=0}^n Y_i \phi_i(x), 
$$

where $\phi_i(x)$ is the $i$-th Lagrange polynomial defined by 
$$
    \phi_i(x) = \frac{(x - X_0)\ldots (x - X_{i-1})(x - X_{i+1})\ldots(x - X_n)}{(X_i - X_0)\ldots (X_i - X_{i-1})(X_i - X_{i+1})\ldots(X_i - X_n)}

$$

```python
def lagrange(x,X,Y):
 
  n = min(len(X),len(Y))
  m = len(x)  
  yh = np.zeros(m)
  phi = np.zeros(n)  
  for j in range(m):
    yh[j] = 0
    for i in range(n):
      phi[i] = 1.0
      for k in range(n):
        if i != k:
          phi[i] = phi[i]*(x[j]-X[k])/(X[i]-X[k])
      yh[j] = yh[j] + Y[i] * phi[i]
  return yh
```




## Visualization of the interpolation


```python
x = np.linspace(-5,5,num=100)
fx = f(x)
y = lagrange(x,X,Y)

fig, ax = plt.subplots()
ax.set_title('Lagrange interpolation of Runge function', fontsize = 15)
ax.plot(x,fx,label='Runge function')
ax.plot(x,y,label='Lagrange interpolation')
plt.legend()
plt.show()
```

<figure><img src="interpolation.png" alt=""><figcaption></figcaption></figure>


## Convert your model to Cairo

Now that we have a good understanding of the Lagrange interpolation, we will replicate the entire algorithm in Cairo to make it fully verifiable. Since we will be rebuilding the algorithm from scratch, this will be a good opportunity to get acquainted with Orion's built-in functions and the operators that make the transition to Cairo seamless.

### Create a new Scarb project

Scarb is the Cairo package manager specifically created to streamline our Cairo and Starknet development process. Scarb will typically manage project dependencies, the compilation process (both pure Cairo and Starknet contracts), downloading and building external libraries to accelerate our development with Orion.You can find all information about Scarb and Cairo installation [here](https://orion.gizatech.xyz/v/develop/framework/get-started#installations).

To create a new Scarb project, open your terminal and run:

```
scarb new verifiable_Lagrange_interpolation
```

A new project folder will be created for you and make sure to replace the content in Scarb.toml file with the following code:

```toml
[package]
name = "scarb new verifiable_Lagrange_interpolation"
version = "0.1.0"

[dependencies]
orion = { git = "https://github.com/gizatechxyz/orion.git", rev = "v0.1.9" }
```

### Gerating the dataset in Cairo

Now let's generate the necessary files to begin our transition to Cairo. In our Jupyter Notebook, we will run the necessary code to convert our interpolation nodes and the interpolation result as fixed point tensors in Orion.

```python
import os
```

```python
os.makedirs("src/generated", exist_ok=True)
```

```python
tensor_name = ["X","Y","x","y"]

def generate_cairo_files(data, name):

    with open(os.path.join('src', 'generated', f"{name}.cairo"), "w") as f:
        f.write(
            "use array::{ArrayTrait, SpanTrait};\n" +
            "use orion::operators::tensor::{core::{Tensor, TensorTrait}};\n" +
            "use orion::operators::tensor::FP16x16Tensor;\n" +
            "use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FixedTrait};\n" +
            "\n" + f"fn {name}() -> Tensor<FP16x16>" + "{\n\n" + 
            "let mut shape = ArrayTrait::new();\n"
        )
        for dim in data.shape:
            f.write(f"shape.append({dim});\n")
    
        f.write("let mut data = ArrayTrait::new();\n")
        for val in np.nditer(data.flatten()):
            f.write(f"data.append(FixedTrait::new({abs(int(decimal_to_fp16x16(val)))}, {str(val < 0).lower()}));\n")
        f.write(
            "let tensor = TensorTrait::<FP16x16>::new(shape.span(), data.span());\n" +
            "return tensor;\n}"
        )

with open(f"src/generated.cairo", "w") as f:
    for n in tensor_name:
        f.write(f"mod {n};\n")

generate_cairo_files(X, "X")
generate_cairo_files(Y, "Y")

generate_cairo_files(x, "x")
generate_cairo_files(y, "y")
```

The X, Y, x, y tensor values will now be generated under `src/generated` directory.

In `src/lib.cairo` replace the content with the following code:

```rust
mod generated;
mod helper;
mod test;
```

This will tell our compiler to include the separate modules listed above during the compilation of our code. We will be covering each module in detail in the following section, but let’s first review the generated folder files.

```rust
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{core::{Tensor, TensorTrait}};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FixedTrait};

fn X() -> Tensor<FP16x16>{

let mut shape = ArrayTrait::new();
shape.append(11);
let mut data = ArrayTrait::new();
data.append(FixedTrait::new(327680, false));
data.append(FixedTrait::new(311642, false));
data.append(FixedTrait::new(265098, false));
data.append(FixedTrait::new(192605, false));
data.append(FixedTrait::new(101258, false));
data.append(FixedTrait::new(0, false));
data.append(FixedTrait::new(101258, true));
data.append(FixedTrait::new(192605, true));
data.append(FixedTrait::new(265098, true));
data.append(FixedTrait::new(311642, true));
data.append(FixedTrait::new(327680, true));
let tensor = TensorTrait::<FP16x16>::new(shape.span(), data.span());
return tensor;
}
```

Since Cairo does not come with built-in fixed points we have to explicitly define it for our X, Y, x, y values. Luckily, this is already implemented in Orion for us as a struct as shown below:


```rust
// Example of a FP16x16.
struct FP16x16 {
    mag: u32,
    sign: bool
}
```

For this tutorial, we will use fixed point numbers FP16x16 where the magnitude represents the absolute value and the boolean indicates whether the number is negative or positive. In a 16x16 fixed-point format, there are 16 bits dedicated to the integer part of the number and 16 bits for the fractional part of the number. This format allows us to work with a wide range of values and a high degree of precision for conducting the Tensor operations. To replicate the Lagrange interpolation function, we will conduct our operations using FP16x16 Tensors which are also represented as a structs in Orion.

```rust
struct Tensor<T> {
    shape: Span<usize>,
    data: Span<T>
}
```

A `Tensor` in Orion takes a shape and a span array of the data.

## Implementing Lagrange interpolation using Orion

At this stage, we will be reproducing the Lagrange interpolation function now that we have generated our X, Y, x, y Fixedpoint Tensors. We will begin by creating a separate file for our Lagrange interpolation function file named `helper.cairo` to host all of our core functions.

### Lagrange interpolation function

```rust
fn lagrange_interpolation(x_interpolated: @Tensor<FP16x16>, X: @Tensor<FP16x16>, Y: @Tensor<FP16x16>) -> Tensor<FP16x16> {

    let n = ((*X).data.len());
    let m = ((*x_interpolated).data.len());

    let mut y_data = ArrayTrait::<FP16x16>::new();
    let mut phi = ArrayTrait::<FP16x16>::new(); 
    let mut j = 0;
    
    loop {
        if j == m {
            break;
        }
        let mut y_j = FixedTrait::new(0,true);
        let mut i = 0;
        loop {
            if i == n {
                break;
            }
            let mut phi_i = FixedTrait::<FP16x16>::new(65536,true);
            let mut k = 0;
            loop {
                if k == n {
                    break;
                }
                if i != k {
                    phi_i = phi_i * (*(*x_interpolated).data.at(j) - *(*X).data.at(k))/( *(*X).data.at(i) -  *(*X).data.at(k))
                }
                k += 1;
            };
            y_j = y_j +  *(*Y).data.at(i) * phi_i;
            i += 1;
        };
        y_data.append(y_j);
        j += 1;

    };

    return TensorTrait::new((*x_interpolated).shape, y_data.span());

}

```


### Testing the model

Now that we have implemented the Lagrange interpolation function, we can finally test it. We begin by creating a new separate test file named `test.cairo` and import all the necessary Orion libraries, including our X, Y, x and y values found in the generated folder. We also import the Lagrange function from the `helper.cairo` file, as we will rely on them to construct the model.

```rust
#[cfg(test)]
mod tests {
    use traits::TryInto;
    use alexandria_data_structures::array_ext::{SpanTraitExt};
    use array::{ArrayTrait, SpanTrait};
    use orion::operators::tensor::{Tensor, TensorTrait};
    use orion::numbers::fixed_point::{core::{FixedTrait}};

    use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorDiv, FP16x16TensorSub};

    use orion::numbers::fixed_point::implementations::fp16x16::core::{
        FP16x16, FP16x16Impl, FP16x16Add, FP16x16AddEq, FP16x16Sub, FP16x16Mul, FP16x16MulEq,
        FP16x16TryIntoU128, FP16x16PartialEq, FP16x16PartialOrd, FP16x16SubEq, FP16x16Neg,
        FP16x16Div, FP16x16IntoFelt252, FP16x16Print
    };

    use lagrange::helper::lagrange_interpolation;
    use lagrange::generated::{X::X, Y::Y, x::x, y::y};

    #[test]
    #[available_gas(99999999999999999)]
    fn lagrange_test() {
        let tol = FixedTrait::<FP16x16>::new(655, false); // 655 is 0.01 = 1e-2
        let max_iter = 500_usize;

        // Nodes :
        let X = X();
        let Y = Y();

        let x = x();
        let y_expected = y();

        let y_actual = lagrange_interpolation(@x, @X, @Y);

        let mut i = 0;
        loop {
            if i == y_expected.data.len() {
                break;
            }
            
            assert(*y_expected.data.at(i) - *y_actual.data.at(i) < tol, 'difference below threshold');
            i += 1;
        }
    }
}

```

Our model will be tested using the `lagrange_test()` function, which will follow these steps:

1. Data retrieval: The function starts by obtaining the X, Y nodes values and x, y test values coming from the generated folder.
2. Lagrange computation : computation of the y-values from the x-axis test values using the lagrange interpolation function on the nodes X and Y.

Finally, we can execute the test file by running `scarb test`

```shell
scarb test
testing lagrange ...
running 1 tests
test lagrange::test::tests::lagrange_test ... ok (gas usage est.: 672062330)
test result: ok. 1 passed; 0 failed; 0 ignored; 0 filtered out;
```

And as we can see our test cases have passed! 👏

If you've made it this far, well done! You are now capable of building verifiable ML models, making them ever more reliable and transparent than ever before. 

We invite the community to join us in forging a future in making AI transparent and reliable resource for all.
