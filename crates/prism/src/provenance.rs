//! Prism's `ProvenanceSource` marker.
//!
//! Migrated to the [`converge_pack::ProvenanceSource`] trait. Public
//! surface is unchanged at call sites:
//! `PRISM_PROVENANCE.proposed_fact(...)` reads the same.
//!
//! The `converge-core` engine emits the uniform `suggestor.execute`
//! tracing span automatically around every `Suggestor::execute`
//! call. Suggestors override `Suggestor::provenance()` to return
//! `PRISM_PROVENANCE.as_str()` so the engine's span carries the
//! right origin.
//!
//! `prism_execution_identity()` continues to expose the static
//! [`converge_pack::ExecutionIdentity`] for any prism payload that
//! wants to carry one (today thin: prism has no native backend).

use converge_pack::{ExecutionIdentity, ProvenanceSource};

/// Marker type identifying prism-emitted facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Prism;

impl ProvenanceSource for Prism {
    fn as_str(&self) -> &'static str {
        "prism"
    }
}

/// Canonical provenance constant for prism.
pub const PRISM_PROVENANCE: Prism = Prism;

/// Static execution identity for any Prism-emitted fact. See
/// [`converge_pack::ExecutionIdentity::unspecified`].
///
/// Prism is closed-form pure Rust with no native backend whose build
/// commit can drift, so `prism@<crate-version>` is the identity. Mirrors
/// Ferrox's `unspecified_solver_identity()`. Re-open when Prism wraps a
/// native runtime or grows a learned-parameter pack.
#[must_use]
pub fn prism_execution_identity() -> ExecutionIdentity {
    ExecutionIdentity::unspecified(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use converge_pack::ContextKey;

    #[test]
    fn provenance_string_is_stable() {
        assert_eq!(PRISM_PROVENANCE.as_str(), "prism");
    }

    #[test]
    fn proposed_fact_uses_canonical_source_string() {
        let fact = PRISM_PROVENANCE.proposed_fact(
            ContextKey::Diagnostic,
            "diagnostic",
            converge_pack::TextPayload::new("content"),
        );
        assert_eq!(fact.provenance(), "prism");
    }
}
