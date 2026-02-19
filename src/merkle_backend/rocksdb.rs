use std::path::{Path, PathBuf};

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

    fn write_node(&self, batch: &mut WriteBatch, node: &StoredNode) -> Result<Hash> {
        let hash = Self::node_hash(node);
        let key = Self::node_key(hash);
        let bytes = serde_json::to_vec(node).map_err(|e| {
            Error::custom(format!("Failed to serialize RocksDB Merkle node: {}", e))
        })?;
        batch.put(key, bytes);
        Ok(hash)
    }

    fn load_node(&self, hash: Hash) -> Result<Option<StoredNode>> {
        if hash == EMPTY_HASH {
            return Ok(None);
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
        let mut root = EMPTY_HASH;
        let entries = Self::container_entries(container)?;
        for (key, value) in entries {
            let path = keypath(key);
            root = self.insert_at(root, 0, key, value, &path, &mut batch)?;
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
    ) -> Result<Hash> {
        if node_hash == EMPTY_HASH {
            return self.write_node(batch, &StoredNode::Leaf { key, value });
        }
        let node = self.load_node(node_hash)?.ok_or_else(|| {
            Error::custom("Invariant violation: non-empty hash resolved to empty node".to_string())
        })?;
        match node {
            StoredNode::Leaf {
                key: existing_key,
                value: existing_value,
            } => {
                if existing_key == key {
                    return self.write_node(batch, &StoredNode::Leaf { key, value });
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
                    &StoredNode::Leaf {
                        key: existing_key,
                        value: existing_value,
                    },
                )?;
                let new_leaf_hash = self.write_node(batch, &StoredNode::Leaf { key, value })?;

                let (mut subtree_left, mut subtree_right) = (old_leaf_hash, new_leaf_hash);
                if !existing_path[divergence] {
                    subtree_left = old_leaf_hash;
                    subtree_right = new_leaf_hash;
                } else {
                    subtree_left = new_leaf_hash;
                    subtree_right = old_leaf_hash;
                }

                let mut subtree_hash = self.write_node(
                    batch,
                    &StoredNode::Branch {
                        left: subtree_left,
                        right: subtree_right,
                    },
                )?;

                for d in (depth..divergence).rev() {
                    subtree_hash = if path[d] {
                        self.write_node(
                            batch,
                            &StoredNode::Branch {
                                left: EMPTY_HASH,
                                right: subtree_hash,
                            },
                        )?
                    } else {
                        self.write_node(
                            batch,
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
                    let new_right = self.insert_at(right, depth + 1, key, value, path, batch)?;
                    self.write_node(
                        batch,
                        &StoredNode::Branch {
                            left,
                            right: new_right,
                        },
                    )
                } else {
                    let new_left = self.insert_at(left, depth + 1, key, value, path, batch)?;
                    self.write_node(
                        batch,
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
    ) -> Result<Option<(RawValue, RawValue)>> {
        if node_hash == EMPTY_HASH {
            return Ok(None);
        }
        let node = self.load_node(node_hash)?.ok_or_else(|| {
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
                    self.descend_for_proof(right, depth + 1, path, siblings)
                } else {
                    siblings.push(right);
                    self.descend_for_proof(left, depth + 1, path, siblings)
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
        let resolved = self.descend_for_proof(root, 0, &keypath(target_key), &mut siblings)?;

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
