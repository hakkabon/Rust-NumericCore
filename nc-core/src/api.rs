//! Versioned public API compatibility metadata shared across NumericCore.

/// Stability of a public capability's API contract.
///
/// This classification does not describe numerical quality: experimental
/// capabilities are still subject to the validation program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ApiStability {
    /// Available for evaluation without a source-compatibility promise.
    Experimental,
    /// Supported, but its representation may still be refined before 1.0.
    Provisional,
    /// Intended to remain source compatible throughout the 1.x series.
    Stable,
}

/// One entry in the machine-readable NumericCore capability manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Capability {
    pub identifier: &'static str,
    pub stability: ApiStability,
    pub contract_version: u32,
}

/// Schema of the capability manifest itself.
pub const API_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// API compatibility epoch being prepared for the 1.0 release.
pub const API_COMPATIBILITY_EPOCH: u32 = 1;

/// Public capability catalog, in deterministic identifier order.
///
/// Identifiers, stability values, and ordering are shared with
/// Swift-NumericCore's `NumericCoreAPI.capabilities`.
pub const CAPABILITIES: &[Capability] = &[
    capability("ampl-modeling", ApiStability::Provisional),
    capability("circuit-analysis", ApiStability::Provisional),
    capability("dae-integration", ApiStability::Provisional),
    capability("dense-linear-algebra", ApiStability::Stable),
    capability("engineering-models", ApiStability::Provisional),
    capability("fem", ApiStability::Provisional),
    capability("global-minlp", ApiStability::Experimental),
    capability("iterative-linear-solvers", ApiStability::Stable),
    capability("lp-qp-milp", ApiStability::Stable),
    capability("multiphysics-design", ApiStability::Experimental),
    capability("nonlinear-equations", ApiStability::Provisional),
    capability("nonlinear-optimization", ApiStability::Stable),
    capability("numerical-validation", ApiStability::Stable),
    capability("sparse-linear-algebra", ApiStability::Stable),
];

const fn capability(identifier: &'static str, stability: ApiStability) -> Capability {
    Capability {
        identifier,
        stability,
        contract_version: 1,
    }
}

/// Look up a capability by its stable identifier.
pub fn capability_by_id(identifier: &str) -> Option<&'static Capability> {
    CAPABILITIES
        .iter()
        .find(|capability| capability.identifier == identifier)
}

/// Return whether a capability exists and optionally meets a stability floor.
pub fn supports(identifier: &str, minimum_stability: Option<ApiStability>) -> bool {
    capability_by_id(identifier).is_some_and(|capability| {
        minimum_stability.map_or(true, |minimum| capability.stability >= minimum)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn epoch_one_manifest_identity_is_frozen() {
        assert_eq!(API_MANIFEST_SCHEMA_VERSION, 1);
        assert_eq!(API_COMPATIBILITY_EPOCH, 1);
        assert_eq!(
            CAPABILITIES
                .iter()
                .map(|capability| capability.identifier)
                .collect::<Vec<_>>(),
            vec![
                "ampl-modeling",
                "circuit-analysis",
                "dae-integration",
                "dense-linear-algebra",
                "engineering-models",
                "fem",
                "global-minlp",
                "iterative-linear-solvers",
                "lp-qp-milp",
                "multiphysics-design",
                "nonlinear-equations",
                "nonlinear-optimization",
                "numerical-validation",
                "sparse-linear-algebra",
            ]
        );
    }

    #[test]
    fn manifest_is_well_formed_and_queryable() {
        let identifiers: Vec<_> = CAPABILITIES.iter().map(|item| item.identifier).collect();
        assert_eq!(
            identifiers.iter().copied().collect::<BTreeSet<_>>().len(),
            identifiers.len()
        );
        assert!(identifiers.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(CAPABILITIES.iter().all(|item| item.contract_version > 0));
        assert!(supports("lp-qp-milp", Some(ApiStability::Stable)));
        assert!(!supports(
            "multiphysics-design",
            Some(ApiStability::Provisional)
        ));
        assert!(!supports("unknown-capability", None));
    }
}
