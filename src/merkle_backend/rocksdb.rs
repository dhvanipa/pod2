use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use plonky2::{
    field::types::Field,
    hash::{
        hash_types::NUM_HASH_OUT_ELTS, hashing::PlonkyPermutation, poseidon::PoseidonPermutation,
    },
};
use rocksdb::{Options, WriteBatch, DB};
use serde::{Deserialize, Serialize};

use crate::{
    backends::plonky2::primitives::merkletree::{keypath, kv_hash, MerkleProof},
    frontend::{Error, Result},
    merkle_backend::MerkleProofBackend,
    middleware::{Hash, NativeOperation, OperationAux, RawValue, TypedValue, Value, EMPTY_HASH, F},
};

const NODE_PREFIX: &[u8] = b"node:";
const STATE_PREFIX: &[u8] = b"state:";

#[derive(Debug)]
pub struct RocksDbMerkleProofBackend {
    db_path: PathBuf,
    db: DB,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum StoredNode {
    Leaf { key: RawValue, value: RawValue },
    Branch { left: Hash, right: Hash },
}

impl RocksDbMerkleProofBackend {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let db_path = path.into();
        let mut options = Options::default();
        options.create_if_missing(true);
        let db = DB::open(&options, &db_path).map_err(|e| {
            Error::custom(format!(
                "Failed to open RocksDB merkle backend at {}: {}",
                db_path.display(),
                e
            ))
        })?;
        Ok(Self { db_path, db })
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    fn container_commitment(container: &Value) -> Result<Hash> {
        match container.typed() {
            TypedValue::Dictionary(d) => Ok(d.commitment()),
            TypedValue::Set(s) => Ok(s.commitment()),
            TypedValue::Array(a) => Ok(a.commitment()),
            _ => Err(Error::custom(format!(
                "Unsupported container type for RocksDB backend: {}",
                container.typed()
            ))),
        }
    }

    fn container_entries(container: &Value) -> Result<Vec<(RawValue, RawValue)>> {
        match container.typed() {
            TypedValue::Dictionary(d) => {
                Ok(d.kvs().iter().map(|(k, v)| (k.raw(), v.raw())).collect())
            }
            TypedValue::Set(s) => Ok(s
                .set()
                .iter()
                .map(|v| {
                    let rv = v.raw();
                    (rv, rv)
                })
                .collect()),
            TypedValue::Array(a) => Ok(a
                .array()
                .iter()
                .enumerate()
                .map(|(i, v)| (RawValue::from(i as i64), v.raw()))
                .collect()),
            _ => Err(Error::custom(format!(
                "Unsupported container type for RocksDB backend: {}",
                container.typed()
            ))),
        }
    }

    fn hash_key(prefix: &[u8], hash: Hash) -> Vec<u8> {
        let mut out = Vec::with_capacity(prefix.len() + 32);
        out.extend_from_slice(prefix);
        out.extend_from_slice(&RawValue::from(hash).to_bytes());
        out
    }

    fn state_key(root: Hash) -> Vec<u8> {
        Self::hash_key(STATE_PREFIX, root)
    }

    fn node_key(hash: Hash) -> Vec<u8> {
        Self::hash_key(NODE_PREFIX, hash)
    }

    fn node_hash(node: &StoredNode) -> Hash {
        match node {
            StoredNode::Leaf { key, value } => kv_hash(key, Some(*value)),
            StoredNode::Branch { left, right } => {
                let input: Vec<F> = [left.0.to_vec(), right.0.to_vec()].concat();
                hash_with_flag(F::TWO, &input)
            }
        }
    }

    fn write_node(
        &self,
        batch: &mut WriteBatch,
        pending_nodes: &mut HashMap<Hash, StoredNode>,
        node: &StoredNode,
    ) -> Result<Hash> {
        let hash = Self::node_hash(node);
        let key = Self::node_key(hash);
        let bytes = serde_json::to_vec(node).map_err(|e| {
            Error::custom(format!("Failed to serialize RocksDB Merkle node: {}", e))
        })?;
        batch.put(key, bytes);
        pending_nodes.insert(hash, node.clone());
        Ok(hash)
    }

    fn load_node(
        &self,
        hash: Hash,
        pending_nodes: &HashMap<Hash, StoredNode>,
    ) -> Result<Option<StoredNode>> {
        if hash == EMPTY_HASH {
            return Ok(None);
        }
        if let Some(node) = pending_nodes.get(&hash) {
            return Ok(Some(node.clone()));
        }
        let key = Self::node_key(hash);
        let maybe = self
            .db
            .get(key)
            .map_err(|e| Error::custom(format!("RocksDB get(node) failed: {}", e)))?;
        let bytes = maybe.ok_or_else(|| {
            Error::custom(format!(
                "Missing node {} in RocksDB Merkle backend. State index may be corrupt.",
                hash
            ))
        })?;
        let node = serde_json::from_slice::<StoredNode>(&bytes).map_err(|e| {
            Error::custom(format!("Failed to deserialize RocksDB Merkle node: {}", e))
        })?;
        Ok(Some(node))
    }

    fn ensure_indexed(&self, container: &Value) -> Result<Hash> {
        let expected_root = Self::container_commitment(container)?;
        let marker = Self::state_key(expected_root);
        let is_indexed = self
            .db
            .get(&marker)
            .map_err(|e| Error::custom(format!("RocksDB get(state) failed: {}", e)))?
            .is_some();
        if is_indexed {
            return Ok(expected_root);
        }

        let mut batch = WriteBatch::default();
        let mut pending_nodes: HashMap<Hash, StoredNode> = HashMap::new();
        let mut root = EMPTY_HASH;
        let entries = Self::container_entries(container)?;
        for (key, value) in entries {
            let path = keypath(key);
            root = self.insert_at(root, 0, key, value, &path, &mut batch, &mut pending_nodes)?;
        }
        if root != expected_root {
            return Err(Error::custom(format!(
                "RocksDB index root mismatch: built {}, expected {}.",
                root, expected_root
            )));
        }
        batch.put(marker, [1u8]);
        self.db
            .write(batch)
            .map_err(|e| Error::custom(format!("RocksDB write(state) failed: {}", e)))?;
        Ok(root)
    }

    fn insert_at(
        &self,
        node_hash: Hash,
        depth: usize,
        key: RawValue,
        value: RawValue,
        path: &[bool],
        batch: &mut WriteBatch,
        pending_nodes: &mut HashMap<Hash, StoredNode>,
    ) -> Result<Hash> {
        if node_hash == EMPTY_HASH {
            return self.write_node(batch, pending_nodes, &StoredNode::Leaf { key, value });
        }
        let node = self.load_node(node_hash, pending_nodes)?.ok_or_else(|| {
            Error::custom("Invariant violation: non-empty hash resolved to empty node".to_string())
        })?;
        match node {
            StoredNode::Leaf {
                key: existing_key,
                value: existing_value,
            } => {
                if existing_key == key {
                    return self.write_node(batch, pending_nodes, &StoredNode::Leaf { key, value });
                }
                let existing_path = keypath(existing_key);
                let mut divergence = depth;
                while divergence < existing_path.len()
                    && existing_path[divergence] == path[divergence]
                {
                    divergence += 1;
                }
                if divergence >= path.len() {
                    return Err(Error::custom(
                        "Failed to insert key into RocksDB Merkle index: full-path collision"
                            .to_string(),
                    ));
                }

                let old_leaf_hash = self.write_node(
                    batch,
                    pending_nodes,
                    &StoredNode::Leaf {
                        key: existing_key,
                        value: existing_value,
                    },
                )?;
                let new_leaf_hash =
                    self.write_node(batch, pending_nodes, &StoredNode::Leaf { key, value })?;

                let (subtree_left, subtree_right) = if !existing_path[divergence] {
                    (old_leaf_hash, new_leaf_hash)
                } else {
                    (new_leaf_hash, old_leaf_hash)
                };

                let mut subtree_hash = self.write_node(
                    batch,
                    pending_nodes,
                    &StoredNode::Branch {
                        left: subtree_left,
                        right: subtree_right,
                    },
                )?;

                for d in (depth..divergence).rev() {
                    subtree_hash = if path[d] {
                        self.write_node(
                            batch,
                            pending_nodes,
                            &StoredNode::Branch {
                                left: EMPTY_HASH,
                                right: subtree_hash,
                            },
                        )?
                    } else {
                        self.write_node(
                            batch,
                            pending_nodes,
                            &StoredNode::Branch {
                                left: subtree_hash,
                                right: EMPTY_HASH,
                            },
                        )?
                    };
                }
                Ok(subtree_hash)
            }
            StoredNode::Branch { left, right } => {
                if depth >= path.len() {
                    return Err(Error::custom(
                        "Failed to insert key into RocksDB Merkle index: depth overflow"
                            .to_string(),
                    ));
                }
                if path[depth] {
                    let new_right =
                        self.insert_at(right, depth + 1, key, value, path, batch, pending_nodes)?;
                    self.write_node(
                        batch,
                        pending_nodes,
                        &StoredNode::Branch {
                            left,
                            right: new_right,
                        },
                    )
                } else {
                    let new_left =
                        self.insert_at(left, depth + 1, key, value, path, batch, pending_nodes)?;
                    self.write_node(
                        batch,
                        pending_nodes,
                        &StoredNode::Branch {
                            left: new_left,
                            right,
                        },
                    )
                }
            }
        }
    }

    fn descend_for_proof(
        &self,
        node_hash: Hash,
        depth: usize,
        path: &[bool],
        siblings: &mut Vec<Hash>,
        pending_nodes: &HashMap<Hash, StoredNode>,
    ) -> Result<Option<(RawValue, RawValue)>> {
        if node_hash == EMPTY_HASH {
            return Ok(None);
        }
        let node = self.load_node(node_hash, pending_nodes)?.ok_or_else(|| {
            Error::custom("Invariant violation: non-empty hash resolved to empty node".to_string())
        })?;
        match node {
            StoredNode::Leaf { key, value } => Ok(Some((key, value))),
            StoredNode::Branch { left, right } => {
                if depth >= path.len() {
                    return Err(Error::custom(
                        "Merkle descent depth overflow in RocksDB backend".to_string(),
                    ));
                }
                if path[depth] {
                    siblings.push(left);
                    self.descend_for_proof(right, depth + 1, path, siblings, pending_nodes)
                } else {
                    siblings.push(right);
                    self.descend_for_proof(left, depth + 1, path, siblings, pending_nodes)
                }
            }
        }
    }
}

impl MerkleProofBackend for RocksDbMerkleProofBackend {
    fn prove_contains(
        &self,
        container: &Value,
        key: &Value,
        contains: bool,
    ) -> Result<OperationAux> {
        let root = self.ensure_indexed(container)?;
        let target_key = key.raw();
        let mut siblings = Vec::new();
        let pending_nodes = HashMap::new();
        let resolved =
            self.descend_for_proof(root, 0, &keypath(target_key), &mut siblings, &pending_nodes)?;

        let proof = if contains {
            match resolved {
                Some((resolved_key, _resolved_value)) if resolved_key == target_key => MerkleProof {
                    existence: true,
                    siblings,
                    other_leaf: None,
                },
                _ => {
                    return Err(Error::custom(format!(
                        "Containment proof requested for key {} but key is not present in indexed root {}.",
                        key, root
                    )))
                }
            }
        } else {
            match resolved {
                Some((resolved_key, _)) if resolved_key == target_key => {
                    return Err(Error::custom(format!(
                        "Non-containment proof requested for key {} but key exists in indexed root {}.",
                        key, root
                    )))
                }
                Some((resolved_key, resolved_value)) => MerkleProof {
                    existence: false,
                    siblings,
                    other_leaf: Some((resolved_key, resolved_value)),
                },
                None => MerkleProof {
                    existence: false,
                    siblings,
                    other_leaf: None,
                },
            }
        };

        Ok(OperationAux::MerkleProof(proof))
    }

    fn prove_state_transition(
        &self,
        _op: NativeOperation,
        _old_container: &Value,
        _key: &Value,
        _value: Option<&Value>,
    ) -> Result<OperationAux> {
        Err(Error::custom(
            "RocksDB backend currently supports Contains/NotContains proof generation only.",
        ))
    }
}

/// Variation of Poseidon hash which takes as input a flag value and 8 field
/// elements. Mirrors the in-memory Merkle tree hash semantics.
fn hash_with_flag(flag: F, inputs: &[F]) -> Hash {
    assert_eq!(
        inputs.len(),
        <PoseidonPermutation<F> as PlonkyPermutation<F>>::RATE
    );

    let mut perm = <PoseidonPermutation<F> as PlonkyPermutation<F>>::new(core::iter::repeat(flag));
    for input_chunk in inputs.chunks(<PoseidonPermutation<F> as PlonkyPermutation<F>>::RATE) {
        perm.set_from_slice(input_chunk, 0);
        perm.permute();
    }

    let mut outputs = Vec::new();
    loop {
        for &item in perm.squeeze() {
            outputs.push(item);
            if outputs.len() == NUM_HASH_OUT_ELTS {
                return Hash(crate::middleware::HashOut::from_vec(outputs).elements);
            }
        }
        perm.permute();
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs, path::PathBuf, time::SystemTime};

    use super::RocksDbMerkleProofBackend;
    use crate::{
        merkle_backend::MerkleProofBackend,
        middleware::{
            containers::{Array, Dictionary, Set},
            Key, OperationAux, Value,
        },
    };

    fn temp_db_path(name: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("pod2_rocksdb_{}_{}", name, ts))
    }

    fn as_merkle_proof(
        aux: OperationAux,
    ) -> crate::backends::plonky2::primitives::merkletree::MerkleProof {
        match aux {
            OperationAux::MerkleProof(pf) => pf,
            _ => panic!("expected OperationAux::MerkleProof"),
        }
    }

    #[test]
    fn rocksdb_backend_dictionary_contains_and_not_contains() {
        let path = temp_db_path("dict");
        let backend = RocksDbMerkleProofBackend::open(&path).expect("open rocksdb");

        let dict = Dictionary::new(HashMap::from([
            (Key::from("a"), Value::from(11)),
            (Key::from("b"), Value::from(22)),
        ]));
        let root = dict.commitment();

        let contains_pf = as_merkle_proof(
            backend
                .prove_contains(&Value::from(dict.clone()), &Value::from("a"), true)
                .expect("contains proof"),
        );
        Dictionary::verify(root, &contains_pf, &Key::from("a"), &Value::from(11))
            .expect("verify inclusion");

        let not_contains_pf = as_merkle_proof(
            backend
                .prove_contains(&Value::from(dict), &Value::from("missing"), false)
                .expect("non-inclusion proof"),
        );
        Dictionary::verify_nonexistence(root, &not_contains_pf, &Key::from("missing"))
            .expect("verify exclusion");

        fs::remove_dir_all(path).ok();
    }

    #[test]
    fn rocksdb_backend_supports_set_and_array_contains() {
        let path = temp_db_path("set_array");
        let backend = RocksDbMerkleProofBackend::open(&path).expect("open rocksdb");

        let set = Set::new([Value::from("x"), Value::from("y")].into_iter().collect());
        let set_root = set.commitment();
        let set_pf = as_merkle_proof(
            backend
                .prove_contains(&Value::from(set.clone()), &Value::from("x"), true)
                .expect("set contains proof"),
        );
        Set::verify(set_root, &set_pf, &Value::from("x")).expect("verify set inclusion");

        let arr = Array::new(vec![Value::from(10), Value::from(20), Value::from(30)]);
        let arr_root = arr.commitment();
        let arr_pf = as_merkle_proof(
            backend
                .prove_contains(&Value::from(arr.clone()), &Value::from(1_i64), true)
                .expect("array contains proof"),
        );
        Array::verify(arr_root, &arr_pf, 1, &Value::from(20)).expect("verify array inclusion");

        fs::remove_dir_all(path).ok();
    }

    #[test]
    fn rocksdb_backend_persists_index_across_reopen() {
        let path = temp_db_path("reopen");
        let dict = Dictionary::new(HashMap::from([(Key::from("k"), Value::from(7))]));
        let root = dict.commitment();

        {
            let backend = RocksDbMerkleProofBackend::open(&path).expect("open rocksdb");
            let _ = backend
                .prove_contains(&Value::from(dict.clone()), &Value::from("k"), true)
                .expect("initial contains proof");
        }
        {
            let backend = RocksDbMerkleProofBackend::open(&path).expect("reopen rocksdb");
            let pf = as_merkle_proof(
                backend
                    .prove_contains(&Value::from(dict), &Value::from("k"), true)
                    .expect("contains proof after reopen"),
            );
            Dictionary::verify(root, &pf, &Key::from("k"), &Value::from(7))
                .expect("verify inclusion");
        }

        fs::remove_dir_all(path).ok();
    }
}
