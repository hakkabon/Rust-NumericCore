use crate::Shape;

/// Memory ordering for a 2-D layout.
///
/// Default is `ColumnMajor` — see `docs/decisions/0001-column-major-layout.md`.
/// This matches BLAS/LAPACK's Fortran heritage, which is what Accelerate
/// calls into directly, so column-major storage avoids a transpose (or a
/// "lie about leading dimension" trick) on every call into the Accelerate
/// backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Order {
    RowMajor,
    ColumnMajor,
}

/// Strides + shape for a dense buffer. Does not own memory — this is pure
/// index arithmetic, shared by owned and borrowed buffers alike.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Shape,
    strides: Vec<usize>,
    order: Order,
}

impl Layout {
    /// Contiguous layout for `shape` in the given order.
    pub fn contiguous(shape: Shape, order: Order) -> Self {
        let strides = Self::contiguous_strides(&shape, order);
        Self { shape, strides, order }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn order(&self) -> Order {
        self.order
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Flat offset into the backing buffer for a logical index.
    ///
    /// Panics on rank mismatch in debug builds; callers on a hot path
    /// (kernels) are expected to have already validated bounds once up
    /// front rather than per-element.
    pub fn offset(&self, index: &[usize]) -> usize {
        debug_assert_eq!(index.len(), self.strides.len(), "index rank mismatch");
        index.iter().zip(&self.strides).map(|(i, s)| i * s).sum()
    }

    fn contiguous_strides(shape: &Shape, order: Order) -> Vec<usize> {
        let dims = shape.dims();
        let mut strides = vec![1usize; dims.len()];
        match order {
            Order::RowMajor => {
                // Last dimension varies fastest.
                for i in (0..dims.len().saturating_sub(1)).rev() {
                    strides[i] = strides[i + 1] * dims[i + 1];
                }
            }
            Order::ColumnMajor => {
                // First dimension varies fastest.
                for i in 1..dims.len() {
                    strides[i] = strides[i - 1] * dims[i - 1];
                }
            }
        }
        strides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_major_matrix_offsets() {
        // 3x2 column-major: column 0 = [0,1,2], column 1 = [3,4,5]
        let layout = Layout::contiguous(Shape::matrix(3, 2), Order::ColumnMajor);
        assert_eq!(layout.offset(&[0, 0]), 0);
        assert_eq!(layout.offset(&[1, 0]), 1);
        assert_eq!(layout.offset(&[0, 1]), 3);
    }

    #[test]
    fn row_major_matrix_offsets() {
        // 3x2 row-major: row 0 = [0,1], row 1 = [2,3], row 2 = [4,5]
        let layout = Layout::contiguous(Shape::matrix(3, 2), Order::RowMajor);
        assert_eq!(layout.offset(&[0, 0]), 0);
        assert_eq!(layout.offset(&[0, 1]), 1);
        assert_eq!(layout.offset(&[1, 0]), 2);
    }
}
