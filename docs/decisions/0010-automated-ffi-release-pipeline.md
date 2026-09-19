# 0010 — Automated FFI release pipeline; remote binaryTarget replaces local vendoring

## Status
Accepted.

## Context
ADR 0009 chose a local-path `NumericCoreFFI` binary target, vendoring
the compiled Rust core and its generated bindings directly into
`Swift-NumericCore`'s git history, explicitly because `Rust-NumericCore`
had no release cadence yet — that ADR's own "What to do before this
changes" section named the exact trigger for revisiting: real,
regular tagged releases.

That trigger arrived: `Rust-NumericCore` gained `release.yml`
(tag-triggered: builds the XCFramework, runs the full test suite,
verifies the `Headers/module.modulemap` layout, computes the SPM
checksum, publishes both the framework and the generated Swift
bindings as release assets) well before this ADR was written — the
missing piece was `Swift-NumericCore` actually consuming those
releases instead of continuing to vendor local copies by hand.

## Decision
Two changes, together:

1. **`Package.swift`'s `NumericCoreFFI` target moves to a remote
   `url:`/`checksum:` pin**, replacing the local `path:` form. No Rust
   toolchain, no `scripts/update-ffi.sh`, no committed binary needed to
   build `Swift-NumericCore` or consume it as a dependency (e.g. from
   `Swift-DataLens`) — SPM downloads and verifies the checksummed asset
   itself.
2. **A cross-repo dispatch pipeline keeps that pin current
   automatically**, rather than requiring a human to notice a new
   Rust-NumericCore release and update the checksum by hand:
   - `Rust-NumericCore`'s `release.yml`, on every `v*.*.*` tag push,
     builds, tests, publishes the release, then sends a
     `repository_dispatch` (`rust-release` event) to
     `Swift-NumericCore`, authenticated with a PAT stored as the
     `SWIFT_NUMERICCORE_DISPATCH_TOKEN` secret (the default
     `GITHUB_TOKEN` cannot dispatch across repositories).
   - `Swift-NumericCore`'s `update-ffi.yml` listens for that dispatch
     (plus a weekly poll as a fallback, and manual
     `workflow_dispatch` for on-demand runs — e.g. right after the very
     first tagged release, without waiting on the poll), downloads the
     release assets, re-verifies the `Headers/module.modulemap` layout,
     computes the checksum locally via `swift package compute-checksum`
     (not parsed from release notes — recomputed from the actual
     downloaded artifact, so a corrupted or tampered asset fails the
     checksum step rather than silently propagating a stale value),
     rewrites `Package.swift`'s binary target via
     `scripts/set_ffi_binary_target.py`, refreshes
     `Sources/NCBindings/Generated/*.swift`, runs `swift build` +
     `swift test`, and opens a PR — never pushes directly to a
     protected branch.

The very first run of the updated `update-ffi.yml` (once
`Rust-NumericCore` has a tagged release) performs the one-time flip:
`Package.swift` moves from local `path:` to remote `url:`/`checksum:`,
and the now-redundant `Frameworks/` directory is removed from the
working tree in that same PR. Every subsequent run is just a
tag/checksum/bindings refresh — the workflow's logic doesn't
distinguish "first flip" from "routine update," which is what makes
`scripts/set_ffi_binary_target.py` idempotent and form-agnostic rather
than a special one-shot migration script.

## Consequences
- `Swift-NumericCore`'s git history keeps the binary blobs already
  committed under the old vendoring approach — this ADR does not
  rewrite history. Removing them (via `git filter-repo`, not a normal
  commit) is a separate, manual, human-executed decision, since it's
  destructive and needs coordination with anyone else holding a clone.
- A consumer resolving `Swift-NumericCore` as a dependency now needs
  network access to GitHub release assets at resolve time (same as any
  SPM binary target with a remote `url:`) — a change from the local-path
  era's "everything's already in the clone" property, traded for no
  longer needing compiled binaries in git history going forward.
- The dispatch token is a real operational dependency: if
  `SWIFT_NUMERICCORE_DISPATCH_TOKEN` expires or is revoked,
  `release.yml`'s notify step degrades gracefully (logs and exits 0
  rather than failing the release), and `update-ffi.yml`'s weekly poll
  still converges within a week — no release is ever silently missed
  forever, just delayed.
- `scripts/set_ffi_binary_target.py`'s block-replace is a regex over
  `Package.swift`'s exact current formatting, not a general Swift
  parser. It's been verified against the real file (both the
  local-path-to-remote and remote-to-remote transitions), but a
  significant reformatting of the surrounding `Package.swift` (e.g. a
  future SwiftPM manifest restructuring) could require updating the
  regex alongside it — a small, contained fix, not a redesign.

## Alternatives considered
- **A human manually updates the checksum after each release.**
  Rejected: exactly the kind of easy-to-forget, easy-to-typo manual
  step this project's automation-where-it-pays-off stance argues
  against — a wrong checksum silently breaks dependency resolution for
  every consumer until someone notices.
- **`Swift-NumericCore` polls only (no dispatch), simpler but slower.**
  Rejected as the *only* mechanism — kept as the fallback specifically
  because it's simpler and needs no cross-repo secret, but a week's
  staleness window as the primary path was judged worse than the small
  operational cost of maintaining one PAT.
- **Direct push to `main` instead of a PR.** Rejected: an automated
  process changing what dependency binary every consumer resolves is
  exactly the kind of change that benefits from a visible review point,
  even if that review is usually just "CI passed, merge" — matches this
  workflow already using `peter-evans/create-pull-request` rather than
  `git push` directly.
