/// Runtime dtype tag.
///
/// The Swift-facing API (`Matrix<T>`) is generic and never needs this —
/// but the C ABI at the FFI boundary is not generic, so every buffer that
/// crosses into/out of Rust carries one of these tags. Keep this enum's
/// discriminants stable once published; `nc-ffi` exposes them as a C enum
/// and downstream Swift code will pattern-match on the raw values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    Float32 = 0,
    Float64 = 1,
    // Reserved, not yet backed by kernels: Complex32 = 2, Complex64 = 3.
}

impl DType {
    pub const fn size_in_bytes(self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float64 => 8,
        }
    }
}
