#!/usr/bin/env bash
# Builds the NumericCoreFFI.xcframework consumed by Swift-NumericCore's
# NCBindings target, plus the UniFFI-generated Swift bindings that must
# ship alongside it as source.
#
# Mirrors the release process already established for the Layout project
# (Rust core -> UniFFI -> Swift-Layout) — see docs/decisions/0008-split-into-two-repos.md
# for why this repo follows the same pattern rather than inventing a new one.
#
# Requires: a macOS host with Xcode, rustup with the four Apple targets
# installed:
#   rustup target add aarch64-apple-ios aarch64-apple-ios-sim \
#                      aarch64-apple-darwin x86_64-apple-darwin
#
# Verified end-to-end on macOS (Xcode 16, uniffi 0.27). If you change
# the UniFFI interface in nc-ffi/src/lib.rs, run this script, then copy
# target/xcframework/bindings/*.swift into
# Swift-NumericCore/Sources/NCBindings/Generated/ (or run
# Swift-NumericCore/scripts/update-ffi.sh, which does both steps).

set -euo pipefail

CRATE_NAME="nc_ffi"
LIB_NAME="libnc_ffi"
OUT_DIR="target/xcframework"
# BINDINGS_DIR holds the generated Swift source (*.swift) that ships as
# source into Swift-NumericCore/Sources/NCBindings/Generated/.
BINDINGS_DIR="${OUT_DIR}/bindings"
# HEADERS_DIR holds only what `xcodebuild -create-xcframework -headers`
# needs: the C header plus the modulemap. Kept separate from
# BINDINGS_DIR so the generated *.swift never ends up inside the
# .xcframework's Headers/ directory (where it does nothing but confuse).
HEADERS_DIR="${OUT_DIR}/headers"

rm -rf "${OUT_DIR}"
mkdir -p "${BINDINGS_DIR}" "${HEADERS_DIR}"

echo "==> Building for macOS (arm64 + x86_64)"
cargo build --release --target aarch64-apple-darwin -p nc-ffi
cargo build --release --target x86_64-apple-darwin -p nc-ffi
mkdir -p "${OUT_DIR}/macos"
lipo -create \
    "target/aarch64-apple-darwin/release/${LIB_NAME}.a" \
    "target/x86_64-apple-darwin/release/${LIB_NAME}.a" \
    -output "${OUT_DIR}/macos/${LIB_NAME}.a"

echo "==> Building for iOS device (arm64)"
cargo build --release --target aarch64-apple-ios -p nc-ffi
mkdir -p "${OUT_DIR}/ios"
cp "target/aarch64-apple-ios/release/${LIB_NAME}.a" "${OUT_DIR}/ios/${LIB_NAME}.a"

echo "==> Building for iOS simulator (arm64)"
cargo build --release --target aarch64-apple-ios-sim -p nc-ffi
mkdir -p "${OUT_DIR}/ios-sim"
cp "target/aarch64-apple-ios-sim/release/${LIB_NAME}.a" "${OUT_DIR}/ios-sim/${LIB_NAME}.a"

echo "==> Generating Swift bindings via uniffi-bindgen"
# Proc-macro-only mode: generate bindings from the compiled library's
# embedded metadata, not from a .udl file.
# NOTE: uniffi-bindgen writes nc_ffi.swift + nc_ffiFFI.h +
# nc_ffiFFI.modulemap all into one directory. We split them below:
# *.swift -> BINDINGS_DIR (Swift source), *.h/*.modulemap -> HEADERS_DIR.
cargo run --release --features bindgen-cli --bin uniffi-bindgen -- generate \
    --library "target/aarch64-apple-darwin/release/${LIB_NAME}.dylib" \
    --language swift \
    --out-dir "${OUT_DIR}/bindgen-out"
mv "${OUT_DIR}/bindgen-out"/*.swift "${BINDINGS_DIR}/"
mv "${OUT_DIR}/bindgen-out"/*.h "${OUT_DIR}/bindgen-out"/*.modulemap "${HEADERS_DIR}/"
rmdir "${OUT_DIR}/bindgen-out"

# SwiftPM/Clang only auto-loads a modulemap named exactly
# `module.modulemap` from a header search path (-I Headers/). UniFFI
# generates `<crate>FFI.modulemap` (here: nc_ffiFFI.modulemap), which
# Clang silently ignores under that name — `canImport(nc_ffiFFI)` then
# evaluates false, the generated Swift loses RustBuffer/RustCallStatus/
# ForeignBytes, and the downstream link fails with undefined symbols
# for every uniffi_/ffi_ entry point. Copy (not rename, so the
# original stays for reference) to the discoverable name:
cp "${HEADERS_DIR}/${CRATE_NAME}FFI.modulemap" "${HEADERS_DIR}/module.modulemap"

echo "==> Assembling XCFramework"
xcodebuild -create-xcframework \
    -library "${OUT_DIR}/macos/${LIB_NAME}.a" -headers "${HEADERS_DIR}" \
    -library "${OUT_DIR}/ios/${LIB_NAME}.a" -headers "${HEADERS_DIR}" \
    -library "${OUT_DIR}/ios-sim/${LIB_NAME}.a" -headers "${HEADERS_DIR}" \
    -output "${OUT_DIR}/NumericCoreFFI.xcframework"

echo "==> Zipping for release upload"
(cd "${OUT_DIR}" && zip -r NumericCoreFFI.xcframework.zip NumericCoreFFI.xcframework)

echo "==> Done."
echo "    Framework: ${OUT_DIR}/NumericCoreFFI.xcframework.zip"
echo "    Bindings:  ${BINDINGS_DIR}/*.swift  (copy into"
echo "               Swift-NumericCore/Sources/NCBindings/Generated/)"
echo
echo "Next: compute the checksum for Package.swift's binaryTarget:"
echo "    swift package compute-checksum ${OUT_DIR}/NumericCoreFFI.xcframework.zip"
