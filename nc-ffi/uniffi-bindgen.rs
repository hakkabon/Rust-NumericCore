//! Entry point for `cargo run --bin uniffi-bindgen -- generate ...`.
//! See `scripts/build-xcframework.sh` for the full invocation used to
//! produce the Swift bindings shipped alongside the XCFramework.
fn main() {
    uniffi::uniffi_bindgen_main()
}
