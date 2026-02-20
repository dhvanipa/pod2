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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(target_arch = "wasm32")]
fn log_stage(msg: &str) {
    log(msg);
}

#[cfg(not(target_arch = "wasm32"))]
fn log_stage(_msg: &str) {}

fn install_panic_hook_once() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        console_error_panic_hook::set_once();
    });
}

fn run_main_pod_points_inner(mock: bool) -> Result<String, Box<dyn std::error::Error>> {
    log_stage(&format!(
        "[pod2/web] run_main_pod_points start mode={}",
        if mock { "mock" } else { "real" }
    ));
    let params = Params::default();

    let mock_prover = MockProver {};
    let real_prover = Prover {};
    let (vd_set, prover): (_, &dyn MainPodProver) = if mock {
        (VDSet::new(&[]), &mock_prover as &dyn MainPodProver)
    } else {
        log_stage("[pod2/web] building DEFAULT_VD_SET (real prover init)");
        ((*DEFAULT_VD_SET).clone(), &real_prover as &dyn MainPodProver)
    };
    log_stage("[pod2/web] prover and vd_set ready");

    let game_sk = SecretKey::new_rand();
    let game_pk = game_sk.public_key();
    let game_signer = Signer(game_sk);

    let mut builder = SignedDictBuilder::new(&params);
    builder.insert("player", "Alice");
    builder.insert("level", 1);
    builder.insert("points", 3512);
    let pod_points_lvl_1 = builder.sign(&game_signer)?;
    pod_points_lvl_1.verify()?;
    log_stage("[pod2/web] signed and verified level-1 dict");

    let mut builder = SignedDictBuilder::new(&params);
    builder.insert("player", "Alice");
    builder.insert("level", 2);
    builder.insert("points", 5771);
    let pod_points_lvl_2 = builder.sign(&game_signer)?;
    pod_points_lvl_2.verify()?;
    log_stage("[pod2/web] signed and verified level-2 dict");

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
    log_stage("[pod2/web] custom predicate module loaded");
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
    log_stage("[pod2/web] proved pod_alice_lvl_1_points");

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
    log_stage("[pod2/web] proved pod_alice_lvl_2_points");

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
    log_stage("[pod2/web] proved and verified pod_alice_over_9000");

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
