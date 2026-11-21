//! Deterministic cut storage for decomposition drivers.

use crate::{Bound, LpRow, LpSubproblemID, OptimizeError};
use std::collections::{BTreeMap, BTreeSet};

pub type CutId = u64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CutScope {
    Global,
    Subproblem(LpSubproblemID),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutOrigin {
    pub source: LpSubproblemID,
    pub iteration: usize,
    pub kind: CutKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutKind {
    Feasibility,
    Optimality,
}

/// A caller-supplied inequality in the form aᵀx ≤ upper.
#[derive(Debug, Clone)]
pub struct DecompositionCut {
    pub entries: Vec<(usize, f64)>,
    pub upper: f64,
    pub origin: CutOrigin,
    pub scope: CutScope,
}

impl DecompositionCut {
    pub fn lp_row(&self) -> LpRow {
        LpRow {
            entries: self.entries.clone(),
            bound: Bound {
                lower: None,
                upper: Some(self.upper),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct CutPoolEntry {
    pub id: CutId,
    pub cut: DecompositionCut,
    pub inactive_rounds: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutInsertion {
    Inserted(CutId),
    Duplicate(CutId),
}

/// Keeps immutable cut IDs in insertion order. Duplicate checking compares
/// exact canonical inequalities within the same scope. The coordinator is
/// responsible for cut derivation and for rebuilding a master after deletion.
#[derive(Debug, Clone, Default)]
pub struct DecompositionCutPool {
    next_id: CutId,
    entries: BTreeMap<CutId, CutPoolEntry>,
}

impl DecompositionCutPool {
    pub fn insert(&mut self, mut cut: DecompositionCut) -> Result<CutInsertion, OptimizeError> {
        if !cut.upper.is_finite() || cut.entries.iter().any(|(_, v)| !v.is_finite()) {
            return Err(invalid_cut());
        }
        cut.entries.sort_by_key(|entry| entry.0);
        if cut.entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(invalid_cut());
        }
        cut.entries.retain(|entry| entry.1 != 0.0);
        if let Some(entry) = self.entries.values_mut().find(|entry| {
            entry.cut.scope == cut.scope
                && entry.cut.upper == cut.upper
                && entry.cut.entries == cut.entries
        }) {
            entry.inactive_rounds = 0;
            return Ok(CutInsertion::Duplicate(entry.id));
        }
        let id = self.next_id;
        self.next_id = id.checked_add(1).ok_or_else(invalid_cut)?;
        self.entries.insert(
            id,
            CutPoolEntry {
                id,
                cut,
                inactive_rounds: 0,
            },
        );
        Ok(CutInsertion::Inserted(id))
    }

    pub fn entries(&self) -> impl Iterator<Item = &CutPoolEntry> {
        self.entries.values()
    }

    pub fn remove(&mut self, id: CutId) -> Option<CutPoolEntry> {
        self.entries.remove(&id)
    }

    /// Update age from the coordinator's active-cut IDs for one round.
    pub fn age(&mut self, active: &BTreeSet<CutId>) {
        for entry in self.entries.values_mut() {
            entry.inactive_rounds = if active.contains(&entry.id) {
                0
            } else {
                entry.inactive_rounds.saturating_add(1)
            };
        }
    }

    /// Remove cuts strictly older than the allowed number of inactive rounds.
    pub fn prune(&mut self, maximum_inactive_rounds: usize) -> Vec<CutId> {
        let mut removed = Vec::new();
        self.entries.retain(|id, entry| {
            if entry.inactive_rounds <= maximum_inactive_rounds {
                true
            } else {
                removed.push(*id);
                false
            }
        });
        removed
    }
}

fn invalid_cut() -> OptimizeError {
    OptimizeError::InvalidProblem("invalid decomposition cut or cut identifier overflow".into())
}
