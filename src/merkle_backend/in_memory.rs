use crate::{
    frontend::{Error, Result},
    merkle_backend::{require_value, MerkleProofBackend},
    middleware::{NativeOperation, OperationAux, Value},
};

#[derive(Debug, Default)]
pub struct InMemoryMerkleProofBackend;

impl MerkleProofBackend for InMemoryMerkleProofBackend {
    fn prove_contains(
        &self,
        container: &Value,
        key: &Value,
        contains: bool,
    ) -> Result<OperationAux> {
        let proof = if contains {
            container.prove_existence(key)?.1
        } else {
            container.prove_nonexistence(key)?
        };
        Ok(OperationAux::MerkleProof(proof))
    }

    fn prove_state_transition(
        &self,
        op: NativeOperation,
        old_container: &Value,
        key: &Value,
        value: Option<&Value>,
    ) -> Result<OperationAux> {
        use NativeOperation::{
            ContainerDeleteFromEntries, ContainerInsertFromEntries, ContainerUpdateFromEntries,
        };
        let proof = match op {
            ContainerInsertFromEntries => old_container
                .prove_insertion(key, require_value(value, "ContainerInsertFromEntries")?)?,
            ContainerUpdateFromEntries => old_container
                .prove_update(key, require_value(value, "ContainerUpdateFromEntries")?)?,
            ContainerDeleteFromEntries => old_container.prove_deletion(key)?,
            _ => {
                return Err(Error::custom(format!(
                    "Unsupported state transition operation {:?} for Merkle proof backend.",
                    op
                )))
            }
        };
        Ok(OperationAux::MerkleTreeStateTransitionProof(proof))
    }
}
