//! Module that implements the MerkleTree specified at
//! <https://0xparc.github.io/pod2/merkletree.html> .
use std::{collections::HashMap, fmt, iter::IntoIterator, sync::Arc};

use itertools::zip_eq;
use plonky2::{
    field::types::Field,
    hash::{
        hash_types::NUM_HASH_OUT_ELTS, hashing::PlonkyPermutation, poseidon::PoseidonPermutation,
    },
};
use serde::{Deserialize, Serialize};

use crate::middleware::{Hash, RawValue, EMPTY_HASH, EMPTY_VALUE, F};

pub mod circuit;
pub use circuit::*;
pub mod error;
pub use error::{TreeError, TreeResult};
pub mod storage;
pub use storage::{
    in_memory_storage, DiskMerkleStorage, InMemoryMerkleStorage, MerkleStorage, StoredNode,
};

/// Theoretical max depth of a merkle tree.  This limits appears because we store keys of 256 bits.
const MAX_DEPTH: usize = 256;

/// Implements the MerkleTree specified at
/// <https://0xparc.github.io/pod2/merkletree.html>
#[derive(Clone, Debug)]
pub struct MerkleTree {
    root: Hash,
    storage: Arc<dyn MerkleStorage>,
}

impl PartialEq for MerkleTree {
    fn eq(&self, other: &Self) -> bool {
        self.root() == other.root()
    }
}
impl Eq for MerkleTree {}

impl MerkleTree {
    /// builds a new `MerkleTree` where the leaves contain the given key-values
    pub fn new(kvs: &HashMap<RawValue, RawValue>) -> Self {
        Self::new_with_storage(kvs, in_memory_storage()).expect("in-memory storage")
    }

    /// builds a new `MerkleTree` using a custom storage backend.
    pub fn new_with_storage(
        kvs: &HashMap<RawValue, RawValue>,
        storage: Arc<dyn MerkleStorage>,
    ) -> TreeResult<Self> {
        let mut tree = Self {
            root: EMPTY_HASH,
            storage,
        };
        tree.storage.save_root(EMPTY_HASH)?;
        for (k, v) in kvs {
            tree.insert_key_value(*k, *v)?;
        }
        Ok(tree)
    }

    /// Loads a `MerkleTree` from the given storage backend snapshot.
    pub fn from_storage(storage: Arc<dyn MerkleStorage>) -> TreeResult<Self> {
        let root = storage.load_root()?.unwrap_or(EMPTY_HASH);
        if root != EMPTY_HASH {
            Self::validate_node_graph(&storage, root)?;
        }
        Ok(Self { root, storage })
    }

    /// returns the root of the tree
    pub fn root(&self) -> Hash {
        self.root
    }

    /// returns the value at the given key
    pub fn get(&self, key: &RawValue) -> TreeResult<RawValue> {
        let path = keypath(*key);
        let (key_resolution, _) = self.down(0, path, None)?;
        match key_resolution {
            Some((k, v)) if &k == key => Ok(v),
            _ => Err(TreeError::key_not_found()),
        }
    }

    /// returns a boolean indicating whether the key exists in the tree
    pub fn contains(&self, key: &RawValue) -> TreeResult<bool> {
        let path = keypath(*key);
        match self.down(0, path, None)? {
            (Some((k, _)), _) if &k == key => Ok(true),
            _ => Ok(false),
        }
    }

    pub fn insert(
        &mut self,
        key: &RawValue,
        value: &RawValue,
    ) -> TreeResult<MerkleTreeStateTransitionProof> {
        let proof_non_existence = self.prove_nonexistence(key)?;
        let old_root = self.root;
        self.insert_key_value(*key, *value)?;
        let new_root = self.root;

        let (v, proof) = self.prove(key)?;
        assert!(proof.existence);
        assert_eq!(v, *value);
        assert!(proof.other_leaf.is_none());

        Ok(MerkleTreeStateTransitionProof {
            op: MerkleTreeOp::Insert, // insertion
            old_root,
            op_proof: proof_non_existence,
            new_root,
            op_key: *key,
            op_value: *value,
            value: None,
            siblings: proof.siblings,
        })
    }

    pub fn update(
        &mut self,
        key: &RawValue,
        value: &RawValue,
    ) -> TreeResult<MerkleTreeStateTransitionProof> {
        let (old_value, old_proof) = self.prove(key)?;
        let old_root = self.root;
        self.update_key_value(*key, *value)?;
        let new_root = self.root;

        let (v, proof) = self.prove(key)?;
        assert!(proof.existence);
        assert_eq!(v, *value);
        assert!(proof.other_leaf.is_none());

        Ok(MerkleTreeStateTransitionProof {
            op: MerkleTreeOp::Update,
            old_root,
            op_proof: old_proof,
            new_root,
            op_key: *key,
            op_value: *value,
            value: Some(old_value),
            siblings: proof.siblings,
        })
    }

    pub fn delete(&mut self, key: &RawValue) -> TreeResult<MerkleTreeStateTransitionProof> {
        let (value, proof_existence) = self.prove(key)?;
        let old_root = self.root;
        self.delete_key(*key)?;
        let new_root = self.root;

        let proof = self.prove_nonexistence(key)?;
        assert!(!proof.existence);

        Ok(MerkleTreeStateTransitionProof {
            op: MerkleTreeOp::Delete,
            old_root,
            op_proof: proof,
            new_root,
            op_key: *key,
            op_value: value,
            value: None,
            siblings: proof_existence.siblings,
        })
    }

    fn set_root(&mut self, new_root: Hash) -> TreeResult<()> {
        self.root = new_root;
        self.storage.save_root(new_root)
    }

    fn load_node_required(&self, hash: Hash) -> TreeResult<StoredNode> {
        self.storage.load_node(hash)?.ok_or_else(|| {
            TreeError::from(anyhow::anyhow!("missing merkle node for hash {}", hash))
        })
    }

    fn save_leaf(&self, key: RawValue, value: RawValue) -> TreeResult<Hash> {
        let hash = kv_hash(&key, Some(value));
        self.storage
            .save_node(hash, &StoredNode::Leaf { key, value })?;
        Ok(hash)
    }

    fn save_intermediate(&self, left: Hash, right: Hash) -> TreeResult<Hash> {
        if left == EMPTY_HASH && right == EMPTY_HASH {
            return Ok(EMPTY_HASH);
        }
        let input: Vec<F> = [left.0.to_vec(), right.0.to_vec()].concat();
        let hash = hash_with_flag(F::TWO, &input);
        self.storage
            .save_node(hash, &StoredNode::Intermediate { left, right })?;
        Ok(hash)
    }

    fn is_intermediate_hash(&self, hash: Hash) -> TreeResult<bool> {
        if hash == EMPTY_HASH {
            return Ok(false);
        }
        Ok(matches!(
            self.load_node_required(hash)?,
            StoredNode::Intermediate { .. }
        ))
    }

    fn validate_node_graph(storage: &Arc<dyn MerkleStorage>, hash: Hash) -> TreeResult<()> {
        if hash == EMPTY_HASH {
            return Ok(());
        }
        let node = storage
            .load_node(hash)?
            .ok_or_else(|| TreeError::from(anyhow::anyhow!("missing merkle node for hash {}", hash)))?;
        match node {
            StoredNode::Leaf { key, value } => {
                let computed = kv_hash(&key, Some(value));
                if computed != hash {
                    return Err(TreeError::from(anyhow::anyhow!(
                        "stored leaf hash mismatch: stored={}, computed={}",
                        hash, computed
                    )));
                }
                Ok(())
            }
            StoredNode::Intermediate { left, right } => {
                let input: Vec<F> = [left.0.to_vec(), right.0.to_vec()].concat();
                let computed = hash_with_flag(F::TWO, &input);
                if computed != hash {
                    return Err(TreeError::from(anyhow::anyhow!(
                        "stored intermediate hash mismatch: stored={}, computed={}",
                        hash, computed
                    )));
                }
                Self::validate_node_graph(storage, left)?;
                Self::validate_node_graph(storage, right)
            }
        }
    }

    fn rebuild_with_siblings(&self, path: &[bool], siblings: &[Hash], mut child_hash: Hash) -> TreeResult<Hash> {
        for i in (0..siblings.len()).rev() {
            let sibling_hash = siblings[i];
            let (left, right) = if path[i] {
                (sibling_hash, child_hash)
            } else {
                (child_hash, sibling_hash)
            };
            child_hash = self.save_intermediate(left, right)?;
        }
        Ok(child_hash)
    }

    fn rebuild_with_siblings_after_delete(
        &self,
        path: &[bool],
        siblings: &[Hash],
        mut child_hash: Hash,
    ) -> TreeResult<Hash> {
        for i in (0..siblings.len()).rev() {
            let sibling_hash = siblings[i];
            let (left, right) = if path[i] {
                (sibling_hash, child_hash)
            } else {
                (child_hash, sibling_hash)
            };
            if left == EMPTY_HASH && !self.is_intermediate_hash(right)? {
                child_hash = right;
                continue;
            }
            if right == EMPTY_HASH && !self.is_intermediate_hash(left)? {
                child_hash = left;
                continue;
            }
            child_hash = self.save_intermediate(left, right)?;
        }
        Ok(child_hash)
    }

    fn build_collision_subtree(
        &self,
        old_key: RawValue,
        old_value: RawValue,
        new_key: RawValue,
        new_value: RawValue,
        lvl: usize,
    ) -> TreeResult<Hash> {
        let old_path = keypath(old_key);
        let new_path = keypath(new_key);
        if lvl >= MAX_DEPTH {
            return Err(TreeError::max_depth());
        }

        if old_path[lvl] != new_path[lvl] {
            let old_hash = self.save_leaf(old_key, old_value)?;
            let new_hash = self.save_leaf(new_key, new_value)?;
            if new_path[lvl] {
                self.save_intermediate(old_hash, new_hash)
            } else {
                self.save_intermediate(new_hash, old_hash)
            }
        } else {
            let next_hash =
                self.build_collision_subtree(old_key, old_value, new_key, new_value, lvl + 1)?;
            if new_path[lvl] {
                self.save_intermediate(EMPTY_HASH, next_hash)
            } else {
                self.save_intermediate(next_hash, EMPTY_HASH)
            }
        }
    }

    fn insert_key_value(&mut self, key: RawValue, value: RawValue) -> TreeResult<()> {
        let path = keypath(key);
        let mut siblings = Vec::new();
        let (terminal, lvl) = self.down(0, path.clone(), Some(&mut siblings))?;
        let replacement_hash = match terminal {
            None => self.save_leaf(key, value)?,
            Some((k, _v)) if k == key => return Err(TreeError::key_exists()),
            Some((k, v)) => self.build_collision_subtree(k, v, key, value, lvl)?,
        };
        let new_root = self.rebuild_with_siblings(&path, &siblings, replacement_hash)?;
        self.set_root(new_root)
    }

    fn update_key_value(&mut self, key: RawValue, value: RawValue) -> TreeResult<()> {
        let path = keypath(key);
        let mut siblings = Vec::new();
        let (terminal, _lvl) = self.down(0, path.clone(), Some(&mut siblings))?;
        match terminal {
            Some((k, _)) if k == key => {
                let replacement_hash = self.save_leaf(key, value)?;
                let new_root = self.rebuild_with_siblings(&path, &siblings, replacement_hash)?;
                self.set_root(new_root)
            }
            _ => Err(TreeError::key_not_found()),
        }
    }

    fn delete_key(&mut self, key: RawValue) -> TreeResult<()> {
        let path = keypath(key);
        let mut siblings = Vec::new();
        let (terminal, _lvl) = self.down(0, path.clone(), Some(&mut siblings))?;
        match terminal {
            Some((k, _)) if k == key => {
                let new_root = self.rebuild_with_siblings_after_delete(
                    &path,
                    &siblings,
                    EMPTY_HASH,
                )?;
                self.set_root(new_root)
            }
            _ => Err(TreeError::key_not_found()),
        }
    }

    /// Traverses the persisted Merkle tree from the stored root toward the
    /// terminal node (leaf or empty branch) for the given key path.
    ///
    /// This method is storage-backed: each step resolves children through
    /// `MerkleStorage::load_node` using node hashes.
    ///
    /// If `siblings` is provided, it is filled with sibling hashes along the
    /// traversed path (from top to bottom), which is used by proof generation.
    ///
    /// The returned leaf (if any) is the leaf reached by path resolution and
    /// may have a different key/value than the queried key in non-existence
    /// cases.
    ///
    /// Missing referenced nodes are treated as storage inconsistency and return
    /// an error.
    fn down(
        &self,
        mut lvl: usize,
        path: Vec<bool>,
        mut siblings: Option<&mut Vec<Hash>>,
    ) -> TreeResult<(Option<(RawValue, RawValue)>, usize)> {
        let root_hash = self.root;
        if root_hash == EMPTY_HASH {
            return Ok((None, lvl));
        }

        let mut cur = root_hash;
        loop {
            let node = self.storage.load_node(cur)?;
            match node {
                Some(StoredNode::Leaf { key, value }) => return Ok((Some((key, value)), lvl)),
                Some(StoredNode::Intermediate { left, right }) => {
                    if path[lvl] {
                        if let Some(s) = siblings.as_mut() {
                            s.push(left);
                        }
                        cur = right;
                    } else {
                        if let Some(s) = siblings.as_mut() {
                            s.push(right);
                        }
                        cur = left;
                    }
                    if cur == EMPTY_HASH {
                        return Ok((None, lvl + 1));
                    }
                    lvl += 1;
                    if lvl > MAX_DEPTH {
                        return Err(TreeError::max_depth());
                    }
                }
                None => {
                    return Err(TreeError::from(anyhow::anyhow!(
                        "missing merkle node for hash {}",
                        cur
                    )));
                }
            }
        }
    }

    /// returns a proof of existence, which proves that the given key exists in
    /// the tree. It returns the `value` of the leaf at the given `key`, and the
    /// `MerkleProof`.
    pub fn prove(&self, key: &RawValue) -> TreeResult<(RawValue, MerkleProof)> {
        let path = keypath(*key);

        let mut siblings: Vec<Hash> = Vec::new();

        match self.down(0, path, Some(&mut siblings))? {
            (Some((k, v)), _) if &k == key => Ok((
                v,
                MerkleProof {
                    existence: true,
                    siblings,
                    other_leaf: None,
                },
            )),
            _ => Err(TreeError::key_not_found()),
        }
    }

    /// returns a proof of non-existence, which proves that the given
    /// `key` does not exist in the tree. The return value specifies
    /// the key-value pair in the leaf reached as a result of
    /// resolving `key` as well as a `MerkleProof`.
    pub fn prove_nonexistence(&self, key: &RawValue) -> TreeResult<MerkleProof> {
        let path = keypath(*key);

        let mut siblings: Vec<Hash> = Vec::new();

        // note: non-existence of a key can be in 2 cases:
        match self.down(0, path, Some(&mut siblings))? {
            // case i) the expected leaf does not exist
            (None, _) => Ok(MerkleProof {
                existence: false,
                siblings,
                other_leaf: None,
            }),
            // case ii) the expected leaf does exist in the tree, but it has a different `key`
            (Some((k, v)), _) if &k != key => Ok(MerkleProof {
                existence: false,
                siblings,
                other_leaf: Some((k, v)),
            }),
            _ => Err(TreeError::key_exists()),
        }
        // both cases prove that the given key don't exist in the tree.
    }

    /// verifies an inclusion proof for the given `key` and `value`
    pub fn verify(
        root: Hash,
        proof: &MerkleProof,
        key: &RawValue,
        value: &RawValue,
    ) -> TreeResult<()> {
        let h = proof.compute_root_from_leaf(key, Some(*value))?;

        if h != root {
            Err(TreeError::proof_fail("inclusion".to_string()))
        } else {
            Ok(())
        }
    }

    /// verifies a non-inclusion proof for the given `key`, that is, the given
    /// `key` does not exist in the tree
    pub fn verify_nonexistence(root: Hash, proof: &MerkleProof, key: &RawValue) -> TreeResult<()> {
        match proof.other_leaf {
            Some((k, _v)) if &k == key => {
                Err(TreeError::invalid_proof("non-existence".to_string()))
            }
            _ => {
                let k = proof.other_leaf.map(|(k, _)| k).unwrap_or(*key);
                let v: Option<RawValue> = proof.other_leaf.map(|(_, v)| v);
                let h = proof.compute_root_from_leaf(&k, v)?;

                if h != root {
                    Err(TreeError::proof_fail("exclusion".to_string()))
                } else {
                    Ok(())
                }
            }
        }
    }

    pub fn verify_state_transition(proof: &MerkleTreeStateTransitionProof) -> TreeResult<()> {
        let mut old_siblings = proof.op_proof.siblings.clone();
        let new_siblings = proof.siblings.clone();

        match proof.op {
            // A deletion is but an insertion subject to a time reversal.
            MerkleTreeOp::Delete => {
                let equivalent_insertion_proof = MerkleTreeStateTransitionProof {
                    op: MerkleTreeOp::Insert,
                    new_root: proof.old_root,
                    old_root: proof.new_root,
                    ..proof.clone()
                };
                Self::verify_state_transition(&equivalent_insertion_proof)
            }
            MerkleTreeOp::Update => {
                // check that for the old_root, (op_key, value) *does* exist in the tree
                Self::verify(
                    proof.old_root,
                    &proof.op_proof,
                    &proof.op_key,
                    &proof.value.unwrap(),
                )?;
                // check that for the new_root, (op_key, op_value) *does* exist in the tree
                Self::verify(
                    proof.new_root,
                    &MerkleProof {
                        existence: true,
                        siblings: proof.siblings.clone(),
                        other_leaf: None,
                    },
                    &proof.op_key,
                    &proof.op_value,
                )?;

                // All siblings should agree
                (proof.siblings == proof.op_proof.siblings)
                    .then_some(())
                    .ok_or(TreeError::state_transition_fail(format!(
                        "Invalid proof of update for key {}: Siblings don't match.",
                        proof.op_key
                    )))
            }
            MerkleTreeOp::Insert => {
                // check that for the old_root, the new_key does not exist in the tree
                Self::verify_nonexistence(proof.old_root, &proof.op_proof, &proof.op_key)?;

                // check that new_siblings verify with the new_root
                Self::verify(
                    proof.new_root,
                    &MerkleProof {
                        existence: true,
                        siblings: new_siblings.clone(),
                        other_leaf: None,
                    },
                    &proof.op_key,
                    &proof.op_value,
                )?;

                // if other_leaf exists, check path divergence
                if let Some((other_key, _)) = proof.op_proof.other_leaf {
                    let old_path = keypath(other_key);
                    let new_path = keypath(proof.op_key);

                    let divergence_lvl: usize =
                        match zip_eq(old_path, new_path).position(|(x, y)| x != y) {
                            Some(d) => d,
                            None => return Err(TreeError::max_depth()),
                        };

                    if divergence_lvl != new_siblings.len() - 1 {
                        return Err(TreeError::state_transition_fail(
                            "paths divergence does not match".to_string(),
                        ));
                    }
                }

                // let d=divergence_level, assert that:
                // 1) old_siblings[i] == new_siblings[i] ∀ i \ {d}
                // 2) at i==d, if old_siblings[i] != new_siblings[i]:
                //     old_siblings[i] == EMPTY_HASH
                //     new_siblings[i] == old_leaf_hash

                // First rule out the case of insertion into empty tree.
                if new_siblings.is_empty() {
                    return (old_siblings.is_empty() && proof.old_root == EMPTY_HASH)
                        .then_some(())
                        .ok_or(TreeError::state_transition_fail(
                            "new tree has no siblings yet old tree is not the empty tree"
                                .to_string(),
                        ));
                }

                let d = new_siblings.len() - 1;
                old_siblings.resize(d + 1, EMPTY_HASH);
                for i in 0..d {
                    if old_siblings[i] != new_siblings[i] {
                        return Err(TreeError::state_transition_fail(
                            "siblings don't match: old[i]!=new[i] ∀ i (except at i==d)".to_string(),
                        ));
                    }
                }
                if old_siblings[d] != new_siblings[d] {
                    if old_siblings[d] != EMPTY_HASH {
                        return Err(TreeError::state_transition_fail(
                            "siblings don't match: old[d]!=empty".to_string(),
                        ));
                    }
                    let k = proof
                .op_proof
                .other_leaf
                .map(|(k, _)| k)
                .ok_or(TreeError::state_transition_fail(
                        "proof.proof_non_existence.other_leaf can not be empty for the case old_siblings[d]!=new_siblings[d]".to_string()
                        ))?;
                    let v: Option<RawValue> = proof.op_proof.other_leaf.map(|(_, v)| v);
                    let old_leaf_hash = kv_hash(&k, v);
                    if new_siblings[d] != old_leaf_hash {
                        return Err(TreeError::state_transition_fail(
                            "siblings don't match: new[d]!=old_leaf_hash".to_string(),
                        ));
                    }
                }
                Ok(())
            }
        }
    }

    fn collect_leaves_from(
        &self,
        node_hash: Hash,
        out: &mut Vec<(RawValue, RawValue)>,
    ) -> TreeResult<()> {
        if node_hash == EMPTY_HASH {
            return Ok(());
        }
        match self.load_node_required(node_hash)? {
            StoredNode::Leaf { key, value } => {
                out.push((key, value));
                Ok(())
            }
            StoredNode::Intermediate { left, right } => {
                self.collect_leaves_from(left, out)?;
                self.collect_leaves_from(right, out)
            }
        }
    }

    fn collect_leaves(&self) -> TreeResult<Vec<(RawValue, RawValue)>> {
        let mut out = Vec::new();
        self.collect_leaves_from(self.root, &mut out)?;
        Ok(out)
    }

    fn write_graphviz_node(&self, f: &mut fmt::Formatter<'_>, hash: Hash) -> fmt::Result {
        if hash == EMPTY_HASH {
            return Ok(());
        }
        let node = self.load_node_required(hash).map_err(|_| fmt::Error)?;
        match node {
            StoredNode::Leaf { key, value } => {
                writeln!(f, "\"{}\" [style=filled]", hash)?;
                writeln!(f, "\"k:{}\\nv:{}\" [style=dashed]", key, value)?;
                writeln!(f, "\"{}\" -> {{ \"k:{}\\nv:{}\" }}", hash, key, value)
            }
            StoredNode::Intermediate { left, right } => {
                let left_id = if left == EMPTY_HASH {
                    let id = format!("\"{}_child_of_{}\"", left, hash);
                    writeln!(f, "{} [label=\"{}\"]", id, left)?;
                    id
                } else {
                    writeln!(f, "\"{}\"", left)?;
                    format!("\"{}\"", left)
                };
                let right_id = if right == EMPTY_HASH {
                    let id = format!("\"{}_child_of_{}\"", right, hash);
                    writeln!(f, "{} [label=\"{}\"]", id, right)?;
                    id
                } else {
                    writeln!(f, "\"{}\"", right)?;
                    format!("\"{}\"", right)
                };
                writeln!(f, "\"{}\" -> {{ {} {} }}", hash, left_id, right_id)?;
                self.write_graphviz_node(f, left)?;
                self.write_graphviz_node(f, right)
            }
        }
    }

    /// returns an iterator over the leaves of the tree
    pub fn iter(&self) -> Iter {
        let items = self
            .collect_leaves()
            .expect("failed to traverse persisted merkle tree");
        Iter {
            inner: items.into_iter(),
        }
    }
}

/// Hash function for key-value pairs. Different branch pair hashes to
/// mitigate fake proofs.
pub fn kv_hash(key: &RawValue, value: Option<RawValue>) -> Hash {
    value
        .map(|v| hash_with_flag(F::ONE, &[key.0.to_vec(), v.0.to_vec()].concat()))
        .unwrap_or(EMPTY_HASH)
}

/// Variation of Poseidon hash which takes as input 1 Goldilock element as a
/// flag, and 8 Goldilocks elements as inputs to the hash. Performs the hashing
/// in a single gate.
/// The function is a fork of
/// [hash_n_to_m_no_pad](https://github.com/0xPolygonZero/plonky2/tree/5d9da5a65bbcba2c66eb29c035090eb2e9ccb05f/plonky2/src/hash/hashing.rs#L30)
/// from plonky2.
fn hash_with_flag(flag: F, inputs: &[F]) -> Hash {
    assert_eq!(
        inputs.len(),
        <PoseidonPermutation<F> as PlonkyPermutation<F>>::RATE
    );

    // this will set `perm` to a  `SPONGE_RATE+SPONGE_CAPACITY` (8+4=12) in our
    // case to a vector of repeated `flag` value. Later at the absorption step,
    // it will fit the inputs values at positions 0-8, keeping the flag values
    // at positions 8-12.
    let mut perm = <PoseidonPermutation<F> as PlonkyPermutation<F>>::new(core::iter::repeat(flag));

    // Absorb all input chunks.
    for input_chunk in inputs.chunks(<PoseidonPermutation<F> as PlonkyPermutation<F>>::RATE) {
        perm.set_from_slice(input_chunk, 0);
        perm.permute();
    }

    // Squeeze until we have the desired number of outputs.
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

impl<'a> IntoIterator for &'a MerkleTree {
    type Item = (RawValue, RawValue);
    type IntoIter = Iter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl fmt::Display for MerkleTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\nPaste in GraphViz (https://dreampuf.github.io/GraphvizOnline/):\n-----"
        )?;
        writeln!(f, "digraph hierarchy {{")?;
        writeln!(f, "node [fontname=Monospace,fontsize=10,shape=box]")?;
        self.write_graphviz_node(f, self.root)?;
        writeln!(f, "\n}}\n-----")
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MerkleProof {
    // note: currently we don't use the `_existence` field, we would use if we merge the methods
    // `verify` and `verify_nonexistence` into a single one
    #[allow(unused)]
    pub(crate) existence: bool,
    pub(crate) siblings: Vec<Hash>,
    // other_leaf is used for non-existence proofs
    pub(crate) other_leaf: Option<(RawValue, RawValue)>,
}

impl fmt::Display for MerkleProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, s) in self.siblings.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", s)?;
        }
        Ok(())
    }
}

impl MerkleProof {
    /// Computes the root of the Merkle tree suggested by a Merkle proof given a
    /// key & value. If a value is not provided, the terminal node is assumed to
    /// be empty.
    fn compute_root_from_leaf(&self, key: &RawValue, value: Option<RawValue>) -> TreeResult<Hash> {
        let path = keypath(*key);
        let h = kv_hash(key, value);
        self.compute_root_from_node(&h, path)
    }
    fn compute_root_from_node(&self, node_hash: &Hash, path: Vec<bool>) -> TreeResult<Hash> {
        let mut h = *node_hash;
        for (i, sibling) in self.siblings.iter().enumerate().rev() {
            let input: Vec<F> = if path[i] {
                [sibling.0, h.0].concat()
            } else {
                [h.0, sibling.0].concat()
            };
            h = hash_with_flag(F::TWO, &input);
        }
        Ok(h)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MerkleClaimAndProof {
    pub root: Hash,
    pub key: RawValue,
    pub value: RawValue,
    pub proof: MerkleProof,
}

impl MerkleClaimAndProof {
    pub fn empty() -> Self {
        Self {
            root: EMPTY_HASH,
            key: EMPTY_VALUE,
            value: EMPTY_VALUE,
            proof: MerkleProof {
                existence: true,
                siblings: vec![],
                other_leaf: None,
            },
        }
    }
    pub fn new(root: Hash, key: RawValue, value: Option<RawValue>, proof: MerkleProof) -> Self {
        Self {
            root,
            key,
            value: value.unwrap_or(EMPTY_VALUE),
            proof,
        }
    }
}

impl fmt::Display for MerkleClaimAndProof {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.proof.fmt(f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MerkleTreeOp {
    Insert = 0,
    Update,
    Delete,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MerkleTreeStateTransitionProof {
    pub(crate) op: MerkleTreeOp,

    pub(crate) old_root: Hash,

    /// Insert: proof of non-existence of the op_key for the old_root
    /// Update: proof of existence of (op_key, value) for the old_root
    /// Delete: proof of non-existence of the op_key for the new_root
    pub(crate) op_proof: MerkleProof,

    pub(crate) new_root: Hash,

    /// Key & value relevant to transition proof. These are the
    /// inserted/updated key-value pair for insertions and updates. For
    /// deletions, these are the key-value pair that is deleted.
    pub(crate) op_key: RawValue,
    pub(crate) op_value: RawValue,

    /// Update: value to be replaced.
    pub(crate) value: Option<RawValue>,

    /// Insert: siblings of inserted (op_key, op_value) leading to new_root
    /// Update: siblings of updated (op_key, op_value) leading to new_root
    /// Delete: siblings of deleted (op_key, op_value) leading to old_root
    pub(crate) siblings: Vec<Hash>,
}

impl MerkleTreeStateTransitionProof {
    /// Value used for padding.
    pub fn empty() -> Self {
        let empty_proof_and_claim = MerkleClaimAndProof::empty();
        Self {
            op: MerkleTreeOp::Insert,
            old_root: empty_proof_and_claim.root,
            op_proof: empty_proof_and_claim.proof,
            new_root: empty_proof_and_claim.root,
            op_key: empty_proof_and_claim.key,
            op_value: empty_proof_and_claim.value,
            value: None,
            siblings: vec![],
        }
    }
}

// NOTE 1: think if maybe the length of the returned vector can be <256
// (8*bytes.len()), so that we can do fewer iterations. For example, if the
// tree.max_depth is set to 20, we just need 20 iterations of the loop, not 256.
// NOTE 2: which approach do we take with keys that are longer than the
// max-depth? ie, what happens when two keys share the same path for more bits
// than the max_depth?
/// returns the path of the given key
pub(crate) fn keypath(k: RawValue) -> Vec<bool> {
    let bytes = k.to_bytes();
    debug_assert_eq!(MAX_DEPTH, bytes.len() * 8);
    (0..MAX_DEPTH)
        .map(|n| bytes[n / 8] & (1 << (n % 8)) != 0)
        .collect()
}

pub struct Iter {
    inner: std::vec::IntoIter<(RawValue, RawValue)>,
}

impl Iterator for Iter {
    type Item = (RawValue, RawValue);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

#[cfg(test)]
pub mod tests {
    use std::cmp::Ordering;

    use itertools::Itertools;

    use super::*;

    #[test]
    fn test_merkletree() -> TreeResult<()> {
        let mut kvs = HashMap::new();
        for i in 0..8 {
            if i == 1 {
                continue;
            }
            kvs.insert(RawValue::from(i), RawValue::from(1000 + i));
        }
        let key = RawValue::from(13);
        let value = RawValue::from(1013);
        kvs.insert(key, value);

        let tree = MerkleTree::new(&kvs);
        // when printing the tree, it should print the same tree as in
        // https://0xparc.github.io/pod2/merkletree.html#example-2
        println!("{}", tree);

        // Inclusion checks
        let (v, proof) = tree.prove(&RawValue::from(13))?;
        assert_eq!(v, RawValue::from(1013));
        println!("{}", proof);

        MerkleTree::verify(tree.root(), &proof, &key, &value)?;

        // Exclusion checks
        let key = RawValue::from(12);
        let proof = tree.prove_nonexistence(&key)?;
        assert_eq!(
            proof.other_leaf.unwrap(),
            (RawValue::from(4), RawValue::from(1004))
        );
        println!("{}", proof);

        MerkleTree::verify_nonexistence(tree.root(), &proof, &key)?;

        let key = RawValue::from(1);
        let proof = tree.prove_nonexistence(&RawValue::from(1))?;
        assert_eq!(proof.other_leaf, None);
        println!("{}", proof);

        MerkleTree::verify_nonexistence(tree.root(), &proof, &key)?;

        // Check iterator
        let collected_kvs: Vec<_> = tree.into_iter().collect::<Vec<_>>();

        // Expected key ordering
        let cmp = |k1, k2| {
            let path1 = keypath(k1);
            let path2 = keypath(k2);

            let first_unequal_bits = std::iter::zip(path1, path2).find(|(b1, b2)| b1 != b2);

            match first_unequal_bits {
                Some((b1, b2)) => {
                    if !b1 & b2 {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                }
                _ => Ordering::Equal,
            }
        };

        let sorted_kvs = kvs
            .iter()
            .sorted_by(|(k1, _), (k2, _)| cmp(**k1, **k2))
            .map(|(k, v)| (*k, *v))
            .collect::<Vec<_>>();

        assert_eq!(collected_kvs, sorted_kvs);

        Ok(())
    }

    #[test]
    fn test_state_transition() -> TreeResult<()> {
        let mut kvs = HashMap::new();
        for i in 0..8 {
            kvs.insert(RawValue::from(i), RawValue::from(1000 + i));
        }

        let mut tree = MerkleTree::new(&kvs);
        let old_root = tree.root();

        // key=37 shares path with key=5, till the level 6, needing 2 extra
        // 'empty' nodes between the original position of key=5 with the new
        // position of key=5 and key=37.
        let key = RawValue::from(37);
        let value = RawValue::from(1037);
        let state_transition_proof = tree.insert(&key, &value)?;

        MerkleTree::verify_state_transition(&state_transition_proof)?;
        assert_eq!(state_transition_proof.old_root, old_root);
        assert_eq!(state_transition_proof.new_root, tree.root());
        assert_eq!(state_transition_proof.op_key, key);
        assert_eq!(state_transition_proof.op_value, value);
        assert_eq!(state_transition_proof.value, None);

        // Deleting this key should yield the old tree, and the proof
        // should be the same (mutatis mutandis).
        let mut tree_with_deleted_key = tree.clone();
        let state_transition_proof1 = tree_with_deleted_key.delete(&key)?;
        MerkleTree::verify_state_transition(&state_transition_proof1)?;
        assert_eq!(
            state_transition_proof1.old_root,
            state_transition_proof.new_root
        );
        assert_eq!(
            state_transition_proof1.new_root,
            state_transition_proof.old_root
        );
        assert_eq!(
            state_transition_proof1.op_key,
            state_transition_proof.op_key
        );
        assert_eq!(
            state_transition_proof1.op_value,
            state_transition_proof.op_value
        );
        assert_eq!(
            state_transition_proof1.op_proof,
            state_transition_proof.op_proof
        );
        assert_eq!(
            state_transition_proof1.siblings,
            state_transition_proof.siblings
        );

        // 2nd part of the test. Add a new leaf
        let mut tree_with_another_leaf = tree.clone();
        let key = RawValue::from(21);
        let value = RawValue::from(1021);
        let state_transition_proof = tree_with_another_leaf.insert(&key, &value)?;

        MerkleTree::verify_state_transition(&state_transition_proof)?;

        // Alternatively add this key with another value then update.
        let value1 = RawValue::from(99);
        tree.insert(&key, &value1)?;
        let state_transition_proof1 = tree.update(&key, &value)?;

        MerkleTree::verify_state_transition(&state_transition_proof1)?;

        // `tree` and `tree_with_another_leaf` should coincide.
        assert_eq!(tree.root(), tree_with_another_leaf.root());

        Ok(())
    }
}
