use crate::{
    backends::plonky2::{
        basetypes::DEFAULT_VD_SET, mainpod::Prover, mock::mainpod::MockProver,
        primitives::ec::schnorr::SecretKey, signer::Signer,
    },
    frontend::{MainPodBuilder, Operation, SignedDictBuilder},
    lang::load_module,
    middleware::{MainPodProver, Params, VDSet},
};
use wasm_bindgen::prelude::*;

fn install_panic_hook_once() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        console_error_panic_hook::set_once();
    });
}

fn run_main_pod_points_inner(mock: bool) -> Result<String, Box<dyn std::error::Error>> {
    let params = Params::default();

    let mock_prover = MockProver {};
    let real_prover = Prover {};
    let (vd_set, prover): (_, &dyn MainPodProver) = if mock {
        (VDSet::new(&[]), &mock_prover as &dyn MainPodProver)
    } else {
        ((*DEFAULT_VD_SET).clone(), &real_prover as &dyn MainPodProver)
    };

    let game_sk = SecretKey::new_rand();
    let game_pk = game_sk.public_key();
    let game_signer = Signer(game_sk);

    let mut builder = SignedDictBuilder::new(&params);
    builder.insert("player", "Alice");
    builder.insert("level", 1);
    builder.insert("points", 3512);
    let pod_points_lvl_1 = builder.sign(&game_signer)?;
    pod_points_lvl_1.verify()?;

    let mut builder = SignedDictBuilder::new(&params);
    builder.insert("player", "Alice");
    builder.insert("level", 2);
    builder.insert("points", 5771);
    let pod_points_lvl_2 = builder.sign(&game_signer)?;
    pod_points_lvl_2.verify()?;

    let input = format!(
        r#"
        points(player, level, points_value, private: points_dict) = AND(
            SignedBy(points_dict, PublicKey({game_pk}))
            Contains(points_dict, "player", player)
            Contains(points_dict, "level", level)
            Contains(points_dict, "points", points_value)
        )

        over_9000(player, private: points_lvl_1, points_lvl_2, points_total) = AND(
            points(player, 1, points_lvl_1)
            points(player, 2, points_lvl_2)
            SumOf(points_total, points_lvl_1, points_lvl_2)
            Gt(points_total, 9000)
        )
    "#
    );
    let module = load_module(&input, "points_module", &params, &[])?;
    let batch = module.batch.clone();
    let points_pred = batch
        .predicate_ref_by_name("points")
        .ok_or_else(|| "missing points predicate".to_string())?;
    let over_9000_pred = batch
        .predicate_ref_by_name("over_9000")
        .ok_or_else(|| "missing over_9000 predicate".to_string())?;

    let mut builder = MainPodBuilder::new(&params, &vd_set);
    let st_signed_by = builder.priv_op(Operation::dict_signed_by(&pod_points_lvl_1))?;
    let st_player = builder.priv_op(Operation::dict_contains(
        pod_points_lvl_1.dict.clone(),
        "player",
        "Alice",
    ))?;
    let st_level = builder.priv_op(Operation::dict_contains(
        pod_points_lvl_1.dict.clone(),
        "level",
        1,
    ))?;
    let st_points = builder.priv_op(Operation::dict_contains(
        pod_points_lvl_1.dict.clone(),
        "points",
        3512,
    ))?;
    let st_points_lvl_1 = builder.pub_op(Operation::custom(
        points_pred.clone(),
        [st_signed_by, st_player, st_level, st_points],
    ))?;
    let pod_alice_lvl_1_points = builder.prove(prover)?;
    pod_alice_lvl_1_points.pod.verify()?;

    let mut builder = MainPodBuilder::new(&params, &vd_set);
    let st_signed_by = builder.priv_op(Operation::dict_signed_by(&pod_points_lvl_2))?;
    let st_player = builder.priv_op(Operation::dict_contains(
        pod_points_lvl_2.dict.clone(),
        "player",
        "Alice",
    ))?;
    let st_level = builder.priv_op(Operation::dict_contains(
        pod_points_lvl_2.dict.clone(),
        "level",
        2,
    ))?;
    let st_points = builder.priv_op(Operation::dict_contains(
        pod_points_lvl_2.dict.clone(),
        "points",
        5771,
    ))?;
    let st_points_lvl_2 = builder.pub_op(Operation::custom(
        points_pred,
        [st_signed_by, st_player, st_level, st_points],
    ))?;
    let pod_alice_lvl_2_points = builder.prove(prover)?;
    pod_alice_lvl_2_points.pod.verify()?;

    let mut builder = MainPodBuilder::new(&params, &vd_set);
    builder.add_pod(pod_alice_lvl_1_points)?;
    builder.add_pod(pod_alice_lvl_2_points)?;
    let st_points_total = builder.priv_op(Operation::sum_of(3512 + 5771, 3512, 5771))?;
    let st_gt_9000 = builder.priv_op(Operation::gt(3512 + 5771, 9000))?;
    let _st_over_9000 = builder.pub_op(Operation::custom(
        over_9000_pred,
        [
            st_points_lvl_1,
            st_points_lvl_2,
            st_points_total,
            st_gt_9000,
        ],
    ))?;
    let pod_alice_over_9000 = builder.prove(prover)?;
    pod_alice_over_9000.pod.verify()?;

    Ok(format!(
        "{{\"mode\":\"{}\",\"result\":\"ok\",\"statement\":\"over_9000(\\\"Alice\\\")\"}}",
        if mock { "mock" } else { "real" }
    ))
}

#[wasm_bindgen]
pub fn run_main_pod_points(mock: bool) -> Result<String, JsValue> {
    install_panic_hook_once();
    run_main_pod_points_inner(mock).map_err(|e| JsValue::from_str(&e.to_string()))
}
