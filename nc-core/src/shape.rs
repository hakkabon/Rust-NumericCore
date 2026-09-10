/// Logical dimensions of a tensor-like object, independent of how it is
/// laid out in memory (that's `Layout`'s job).
///
/// v1 is deliberately restricted to rank 1 (`Vector`) and rank 2 (`Matrix`)
/// use, but is expressed as a `Vec<usize>` so `nc-sparse` and a future
/// N-dimensional array type are not blocked later.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    pub fn scalar() -> Self {
        Self(Vec::new())
    }

    pub fn vector(len: usize) -> Self {
        Self(vec![len])
    }

    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self(vec![rows, cols])
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Total element count — product of all dimensions (1 for a scalar,
    /// the empty product).
    pub fn element_count(&self) -> usize {
        if self.0.is_empty() {
            1
        } else {
            self.0.iter().product()
        }
    }

    pub fn rows(&self) -> Option<usize> {
        self.0.first().copied()
    }

    pub fn cols(&self) -> Option<usize> {
        self.0.get(1).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_element_count() {
        assert_eq!(Shape::matrix(3, 4).element_count(), 12);
    }

    #[test]
    fn vector_element_count() {
        assert_eq!(Shape::vector(5).element_count(), 5);
    }
}
