//! 2×2 dense matrix operating on raw f64.
//!
//! Layout convention: row-major
//!   M = ┌ a  b ┐
//!       └ c  d ┘

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Block2x2 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

#[cfg(test)]
mod tests {
    // Tests added in subsequent tasks.
}
