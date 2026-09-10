//! Pure-Rust fallback kernels.
//!
//! These implementations exist for three reasons, in priority order:
//! 1. Correctness oracle — the backend-diffing test harness compares
//!    Accelerate/MPS output against these naive implementations.
//! 2. Coverage for dtypes/platforms Accelerate doesn't reach.
//! 3. A path to non-Mac targets later (Linux/tanix) without a rewrite.
//!
//! Nothing here is expected to be competitive with Accelerate on
//! Apple Silicon — do not spend optimization effort here beyond what
//! `nc-kernels-simd` already buys for free via autovectorization.

/// Minimal numeric trait covering just what these kernels need.
/// Deliberately not `num_traits::Float` (yet) — kept local until a real
/// need for a broader numeric hierarchy shows up.
pub trait Scalar: Copy + Default + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> {
    fn sqrt(self) -> Self;
}

impl Scalar for f32 {
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl Scalar for f64 {
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

/// Column-major dense matmul: `c[m x n] = a[m x k] * b[k x n]`.
///
/// Naive triple loop, ikj order for cache-friendlier access on
/// column-major storage. This is the fallback path only — see module docs.
pub fn matmul<T: Scalar>(a: &[T], b: &[T], c: &mut [T], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k, "a length mismatch");
    assert_eq!(b.len(), k * n, "b length mismatch");
    assert_eq!(c.len(), m * n, "c length mismatch");

    for v in c.iter_mut() {
        *v = T::default();
    }

    // Column-major: element (i, j) of an m x n matrix is at i + j*m.
    for j in 0..n {
        for p in 0..k {
            let b_pj = b[p + j * k];
            for i in 0..m {
                c[i + j * m] = c[i + j * m] + a[i + p * m] * b_pj;
            }
        }
    }
}

/// `y = alpha * x + y`
pub fn axpy<T: Scalar>(alpha: T, x: &[T], y: &mut [T]) {
    assert_eq!(x.len(), y.len(), "axpy: length mismatch");
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi = alpha * *xi + *yi;
    }
}

/// Dot product of two equal-length vectors.
pub fn dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    assert_eq!(x.len(), y.len(), "dot: length mismatch");
    let mut acc = T::default();
    for (xi, yi) in x.iter().zip(y.iter()) {
        acc = acc + *xi * *yi;
    }
    acc
}

/// Euclidean (L2) norm.
pub fn norm2<T: Scalar>(x: &[T]) -> T {
    dot(x, x).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        // 2x2 identity times [1,2,3,4] (column-major) == itself.
        let identity = [1.0, 0.0, 0.0, 1.0];
        let a = [1.0, 2.0, 3.0, 4.0];
        let mut c = [0.0; 4];
        matmul(&identity, &a, &mut c, 2, 2, 2);
        assert_eq!(c, a);
    }

    #[test]
    fn dot_matches_hand_computation() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn norm2_matches_hand_computation() {
        assert_eq!(norm2(&[3.0, 4.0]), 5.0);
    }

    #[test]
    fn axpy_updates_in_place() {
        let x = [1.0, 1.0, 1.0];
        let mut y = [1.0, 2.0, 3.0];
        axpy(2.0, &x, &mut y);
        assert_eq!(y, [3.0, 4.0, 5.0]);
    }
}
