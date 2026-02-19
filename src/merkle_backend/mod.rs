use std::fmt;

use crate::{
    frontend::{Error, Result},
    middleware::{NativeOperation, OperationAux, Value},
};

mod in_memory;
pub use in_memory::InMemoryMerkleProofBackend;

/// Backend for generating Merkle-related operation auxiliary proofs.
pub trait MerkleProofBackend: fmt::Debug + Send + Sync {
    /// Generate aux data for Contains/NotContains operations.
    fn prove_contains(
        &self,
        container: &Value,
        key: &Value,
        contains: bool,
    ) -> Result<OperationAux>;

    /// Generate aux data for container state transition operations.
    fn prove_state_transition(
        &self,
        op: NativeOperation,
        old_container: &Value,
        key: &Value,
        value: Option<&Value>,
    ) -> Result<OperationAux>;
}

pub(crate) fn require_value<'a>(value: Option<&'a Value>, ctx: &str) -> Result<&'a Value> {
    value.ok_or_else(|| Error::custom(format!("Missing value argument for {} operation.", ctx)))
}
