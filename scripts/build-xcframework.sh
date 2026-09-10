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
# This script is a scaffold — it has not been run end-to-end (no macOS
# host was available while writing it). Treat the exact xcodebuild /
# uniffi-bindgen invocations as a starting point to debug against, not
# as verified-working commands.

set -euo pipefail

CRATE_NAME="nc_ffi"
LIB_NAME="libnc_ffi"
OUT_DIR="target/xcframework"
BINDINGS_DIR="${OUT_DIR}/bindings"

rm -rf "${OUT_DIR}"
mkdir -p "${BINDINGS_DIR}"

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
cargo run --release --features bindgen-cli --bin uniffi-bindgen -- generate \
    --library "target/aarch64-apple-darwin/release/${LIB_NAME}.dylib" \
    --language swift \
    --out-dir "${BINDINGS_DIR}"

echo "==> Assembling XCFramework"
xcodebuild -create-xcframework \
    -library "${OUT_DIR}/macos/${LIB_NAME}.a" -headers "${BINDINGS_DIR}" \
    -library "${OUT_DIR}/ios/${LIB_NAME}.a" -headers "${BINDINGS_DIR}" \
    -library "${OUT_DIR}/ios-sim/${LIB_NAME}.a" -headers "${BINDINGS_DIR}" \
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
