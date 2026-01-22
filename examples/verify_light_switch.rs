#![allow(clippy::uninlined_format_args)] // TODO: Remove this in another PR
//! Example of building main pods that verify signed pods and other main pods using custom
//! predicates
//!
//! The example follows a scenario where a game issues signed pods to players with the points
//! accumulated after finishing each game level.  Then we build a custom predicate to prove that
//! the sum of points from level 1 and 2 for a player is over 9000.
//!
//! Run in real mode: `cargo run --release --example main_pod_points`
//! Run in mock mode: `cargo run --release --example main_pod_points -- --mock`
use std::env;

use pod2::{
    backends::plonky2::{
        basetypes::DEFAULT_VD_SET, mainpod::Prover, mock::mainpod::MockProver,
        primitives::ec::schnorr::SecretKey, signer::Signer,
    },
    frontend::{MainPodBuilder, Operation, SignedDictBuilder},
    lang::parse,
    middleware::{hash_values, MainPodProver, Params, VDSet, Value},
};
use serde::Serialize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args: Vec<String> = env::args().collect();
    let mock = args.get(1).is_some_and(|arg1| arg1 == "--mock");
    if mock {
        println!("Using MockMainPod")
    } else {
        println!("Using MainPod")
    }

    // Read pod from file and verify
    let file = std::fs::File::open("modified_pod_light_switch.json")?;
    let pod_light_switch_from_file: pod2::frontend::MainPod = serde_json::from_reader(file)?;
    pod_light_switch_from_file.pod.verify().unwrap();

    Ok(())
}
