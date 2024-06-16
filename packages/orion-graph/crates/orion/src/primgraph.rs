use std::collections::HashMap;

#[derive(Debug)]
pub struct PrimGraph {
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub nodes: Vec<PrimNode>,
    pub initializers: Vec<Initializer>,
    pub shape: HashMap<String, Vec<usize>>,
}

#[derive(Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub raw_data: Vec<u8>,
    pub dtype: Dtype,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            shape: self.shape.clone(),
            raw_data: self.raw_data.clone(),
            dtype: self.dtype.clone(),
        }
    }
}

#[derive(Debug)]
pub struct PrimNode {
    pub name: String,
    pub optype: Primops,
    pub inputs: Vec<String>,
    pub attributes: Vec<Attribute>,
    pub outputs: Vec<String>,
}

#[derive(Debug)]
pub enum Primops {
    Log2,
    Exp2,
    Sin,
    Sqrt,
    Recip,
    Add,
    Mul,
    Mod,
    LessThan,
    SumReduce,
    MaxReduce,
    Reshape,
}

#[derive(Debug, Clone)]
pub struct Initializer {
    pub name: String,
    pub tensor: Tensor,
}

#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub value: bool,
}

#[derive(Debug, Clone)]
pub struct Input {
    pub name: String,
    pub dtype: Dtype,
    pub dim: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Output {
    pub name: String,
    pub dtype: Dtype,
    pub dim: Vec<u32>,
}

#[derive(Debug, Clone)]
pub enum Dtype {
    Double,
    I32,
    U32,
    Bool,
}
