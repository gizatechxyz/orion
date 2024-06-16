use orion::primgraph::{Attribute, Initializer, Input, Output};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CairoGraph {
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub nodes: Vec<CairoNode>,
    pub initializers: Vec<Initializer>,
    pub shape: HashMap<String, Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct CairoNode {
    pub name: String,
    pub optype: CairoOps,
    pub inputs: Vec<String>,
    pub attributes: Vec<Attribute>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CairoOps {
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
