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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args: Vec<String> = env::args().collect();
    let mock = args.get(1).is_some_and(|arg1| arg1 == "--mock");
    if mock {
        println!("Using MockMainPod")
    } else {
        println!("Using MainPod")
    }

    let params = Params::default();

    let mock_prover = MockProver {};
    let real_prover = Prover {};
    let (vd_set, prover): (_, &dyn MainPodProver) = if mock {
        (&VDSet::new(&[]), &mock_prover)
    } else {
        println!("Prebuilding circuits to calculate vd_set...");
        let vd_set = &*DEFAULT_VD_SET;
        println!("vd_set calculation complete");
        (vd_set, &real_prover)
    };

    // Create a schnorr key pair to sign pods
    let game_sk = SecretKey::new_rand();

    let game_signer = Signer(game_sk);

    let light_switch_predicate = r#"
        LightSwitch_base(new_state_hash, private: action, mid_state, old_state, new_state) = AND(
            HashOf(new_state_hash, new_state, 0)
            // Equal(old_state.position, "")
            Equal(old_state.secret, 0)
            DictUpdate(mid_state, old_state, "position", action.position)
            DictUpdate(new_state, mid_state, "secret", action.secret)
            Equal(action.type, "base")
        )
    "#;
    // Build a signed pod
    let mut old_state: SignedDictBuilder = SignedDictBuilder::new(&params);
    old_state.insert("position", "");
    old_state.insert("secret", 0);
    let old_state = old_state.sign(&game_signer)?;
    old_state.verify()?;

    let mut mid_state: SignedDictBuilder = SignedDictBuilder::new(&params);
    mid_state.insert("position", "on");
    mid_state.insert("secret", 0);
    let mid_state = mid_state.sign(&game_signer)?;
    mid_state.verify()?;

    let mut new_state: SignedDictBuilder = SignedDictBuilder::new(&params);
    new_state.insert("position", "on");
    new_state.insert("secret", 42);
    let new_state = new_state.sign(&game_signer)?;
    new_state.verify()?;

    let mut action: SignedDictBuilder = SignedDictBuilder::new(&params);
    action.insert("position", "on");
    action.insert("secret", 42);
    action.insert("type", "base");
    let action = action.sign(&game_signer)?;
    action.verify()?;

    // let old_state_hash = hash_values(&[Value::from(old_state.dict.clone()), Value::from(0)]);
    let new_state_hash = hash_values(&[Value::from(new_state.dict.clone()), Value::from(0)]);

    println!("# old_state:\n{}", old_state);
    println!("# mid_state:\n{}", mid_state);
    println!("# new_state:\n{}", new_state);
    println!("# action:\n{}", action);

    let mut builder = MainPodBuilder::new(&params, vd_set);
    // let st_old_state_hash = builder.pub_op(Operation::hash_of(
    //     old_state_hash,
    //     old_state.dict.clone(),
    //     0,
    // ))?;
    let st_new_state_hash = builder.priv_op(Operation::hash_of(
        new_state_hash,
        new_state.dict.clone(),
        0,
    ))?;

    let st_equal_position = builder.priv_op(Operation::eq((&old_state, "position"), ""))?;
    let st_equal_secret = builder.priv_op(Operation::eq((&old_state, "secret"), 0))?;

    let st_dict_update1 = builder.priv_op(Operation::dict_update(
        mid_state.dict.clone(),
        old_state.dict.clone(),
        "position",
        (&action, "position"),
    ))?;

    let st_dict_update2 = builder.priv_op(Operation::dict_update(
        new_state.dict.clone(),
        mid_state.dict.clone(),
        "secret",
        (&action, "secret"),
    ))?;
    let st_equal_action_type = builder.priv_op(Operation::eq((&action, "type"), "base"))?;
    let light_switch_batch = parse(light_switch_predicate, &params, &[])?.custom_batch;
    let light_switch_pred = light_switch_batch
        .predicate_ref_by_name("LightSwitch_base")
        .unwrap();
    let _st_light_switch = builder.pub_op(Operation::custom(
        light_switch_pred,
        [
            // st_old_state_hash,
            st_new_state_hash,
            // st_equal_position,
            st_equal_secret,
            st_dict_update1,
            st_dict_update2,
            st_equal_action_type,
        ],
    ))?;
    println!("Proving pod_light_switch...");
    let pod_light_switch = builder.prove(prover).unwrap();
    println!("# pod_light_switch\n:{}", pod_light_switch);
    pod_light_switch.pod.verify().unwrap();

    Ok(())
}
