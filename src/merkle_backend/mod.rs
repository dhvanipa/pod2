use std::fmt;

use crate::{
    frontend::{Error, Result},
    middleware::{NativeOperation, OperationAux, Value},
};

mod in_memory;
pub use in_memory::InMemoryMerkleProofBackend;

#[cfg(feature = "merkle_rocksdb")]
mod rocksdb;
#[cfg(feature = "merkle_rocksdb")]
pub use rocksdb::RocksDbMerkleProofBackend;

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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{InMemoryMerkleProofBackend, MerkleProofBackend};
    use crate::middleware::{containers::Dictionary, Key, NativeOperation, OperationAux, Value};

    fn assert_contains_contract(
        backend: &dyn MerkleProofBackend,
        dict: &Dictionary,
        key: &str,
        value: i64,
    ) {
        let aux = backend
            .prove_contains(&Value::from(dict.clone()), &Value::from(key), true)
            .expect("contains proof");
        let pf = match aux {
            OperationAux::MerkleProof(pf) => pf,
            _ => panic!("Expected OperationAux::MerkleProof"),
        };
        Dictionary::verify(dict.commitment(), &pf, &Key::from(key), &Value::from(value))
            .expect("verify inclusion");
    }

    fn assert_not_contains_contract(
        backend: &dyn MerkleProofBackend,
        dict: &Dictionary,
        key: &str,
    ) {
        let aux = backend
            .prove_contains(&Value::from(dict.clone()), &Value::from(key), false)
            .expect("not-contains proof");
        let pf = match aux {
            OperationAux::MerkleProof(pf) => pf,
            _ => panic!("Expected OperationAux::MerkleProof"),
        };
        Dictionary::verify_nonexistence(dict.commitment(), &pf, &Key::from(key))
            .expect("verify exclusion");
    }

    fn assert_insert_contract(
        backend: &dyn MerkleProofBackend,
        dict: &Dictionary,
        key: &str,
        value: i64,
    ) {
        let aux = backend
            .prove_state_transition(
                NativeOperation::ContainerInsertFromEntries,
                &Value::from(dict.clone()),
                &Value::from(key),
                Some(&Value::from(value)),
            )
            .expect("insert transition proof");
        let pf = match aux {
            OperationAux::MerkleTreeStateTransitionProof(pf) => pf,
            _ => panic!("Expected OperationAux::MerkleTreeStateTransitionProof"),
        };
        Dictionary::verify_state_transition(&pf).expect("verify state transition");
    }

    #[test]
    fn in_memory_backend_contracts() {
        let backend = InMemoryMerkleProofBackend;

        let dict = Dictionary::new(HashMap::from([
            (Key::from("a"), Value::from(1)),
            (Key::from("b"), Value::from(2)),
        ]));
        assert_contains_contract(&backend, &dict, "a", 1);
        assert_not_contains_contract(&backend, &dict, "missing");

        let empty = Dictionary::new(HashMap::new());
        assert_insert_contract(&backend, &empty, "new", 42);
    }
}
