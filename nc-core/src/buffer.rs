use crate::{CoreError, Layout, Result};

/// Owned, aligned, strided storage for a single dtype `T`.
///
/// `Buffer` owns its memory (unlike `Layout`, which is pure index math).
/// Borrowed/foreign-owned variants (e.g. memory owned by Swift/Accelerate
/// and merely viewed from Rust) are expected to live in `nc-ffi` as a
/// separate type — keeping ownership semantics out of this core type
/// keeps `nc-core` simple and keeps `unsafe` confined to the FFI crate.
#[derive(Debug, Clone)]
pub struct Buffer<T> {
    data: Vec<T>,
    layout: Layout,
}

impl<T: Clone + Default> Buffer<T> {
    /// Zero-filled buffer matching `layout`'s element count.
    pub fn zeros(layout: Layout) -> Self {
        let n = layout.shape().element_count();
        Self { data: vec![T::default(); n], layout }
    }
}

impl<T> Buffer<T> {
    pub fn from_vec(data: Vec<T>, layout: Layout) -> Result<Self> {
        let expected = layout.shape().element_count();
        if data.len() != expected {
            return Err(CoreError::BufferLengthMismatch {
                shape: layout.shape().clone(),
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, layout })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn get(&self, index: &[usize]) -> &T {
        &self.data[self.layout.offset(index)]
    }

    pub fn get_mut(&mut self, index: &[usize]) -> &mut T {
        let offset = self.layout.offset(index);
        &mut self.data[offset]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Order, Shape};

    #[test]
    fn zeros_has_correct_length() {
        let layout = Layout::contiguous(Shape::matrix(2, 3), Order::ColumnMajor);
        let buf: Buffer<f64> = Buffer::zeros(layout);
        assert_eq!(buf.as_slice().len(), 6);
    }

    #[test]
    fn from_vec_rejects_wrong_length() {
        let layout = Layout::contiguous(Shape::matrix(2, 3), Order::ColumnMajor);
        let result = Buffer::from_vec(vec![0.0; 5], layout);
        assert!(result.is_err());
    }

    #[test]
    fn get_set_roundtrip() {
        let layout = Layout::contiguous(Shape::matrix(2, 2), Order::ColumnMajor);
        let mut buf: Buffer<f64> = Buffer::zeros(layout);
        *buf.get_mut(&[1, 0]) = 42.0;
        assert_eq!(*buf.get(&[1, 0]), 42.0);
    }
}
