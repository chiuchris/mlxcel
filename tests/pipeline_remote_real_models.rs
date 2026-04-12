mod common;

use std::net::TcpListener;
use std::time::Duration;

use common::repo_model_dir;
use mlxcel::distributed::TransportBackend;
use mlxcel::distributed::pipeline::{
    RemotePipelineRuntime, RemotePipelineRuntimeConfig, RemoteStageResponse,
    RemoteStageServiceConfig, RemoteStageServiceHandle, resolve_in_process_pipeline_num_layers,
    resolve_in_process_stage_assignments,
};
use mlxcel::{LanguageModel, distributed::pipeline::PipelineModelRuntime};
use mlxcel_core::cache::SequenceId;

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

fn spawn_llama_remote_runtime(
    model_dir: &std::path::Path,
) -> (
    RemotePipelineRuntime,
    RemoteStageServiceHandle,
    RemoteStageServiceHandle,
) {
    let num_layers = resolve_in_process_pipeline_num_layers(model_dir).unwrap();
    let assignments =
        resolve_in_process_stage_assignments(num_layers, None, Some("0-7,8-15")).unwrap();

    let stage0_addr = format!("127.0.0.1:{}", reserve_port());
    let stage1_addr = format!("127.0.0.1:{}", reserve_port());

    let stage1 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.to_path_buf(),
        bind_address: stage1_addr.clone(),
        stage_assignment: assignments[1].clone(),
        num_stages: 2,
        upstream_peer: Some(stage0_addr.clone()),
        downstream_peer: None,
    })
    .unwrap();
    let stage0 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.to_path_buf(),
        bind_address: stage0_addr.clone(),
        stage_assignment: assignments[0].clone(),
        num_stages: 2,
        upstream_peer: None,
        downstream_peer: Some(stage1_addr.clone()),
    })
    .unwrap();

    assert_eq!(stage0.local_addr(), stage0_addr);
    assert_eq!(stage1.local_addr(), stage1_addr);

    let runtime = RemotePipelineRuntime::new(RemotePipelineRuntimeConfig {
        stage_peers: vec![stage0_addr, stage1_addr],
        transport_backend: TransportBackend::Tcp,
        bind_address: "127.0.0.1:0".to_string(),
        stage_timeout: Duration::from_secs(5),
    })
    .unwrap();

    (runtime, stage0, stage1)
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote pipeline stages"]
fn pipeline_remote_runtime_llama_real_model_parity_and_cleanup() {
    let model_dir = repo_model_dir("llama-3.2-1b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let (model, _) = mlxcel::load_model(&model_dir).unwrap();
    let (runtime, stage0, stage1) = spawn_llama_remote_runtime(&model_dir);

    let prompt_ids = mlxcel_core::from_slice_i32(&[128000, 9906], &[1, 2]);
    let decode_ids = mlxcel_core::from_slice_i32(&[13], &[1, 1]);

    let mut full_caches = model.make_caches();
    let full_prefill = model.forward(&prompt_ids, &mut full_caches, None);
    let full_decode = model.forward(&decode_ids, &mut full_caches, None);

    let seq_id = SequenceId::from_raw(42);
    runtime.prepare_sequence_state(seq_id);
    let remote_prefill = runtime.forward_sequence(seq_id, &prompt_ids, None).unwrap();
    let remote_decode = runtime.forward_sequence(seq_id, &decode_ids, None).unwrap();

    let active_states = runtime.probe_stages().unwrap();
    assert!(active_states.iter().all(|state| matches!(
        state,
        RemoteStageResponse::State {
            state,
            pending_entry_replies: _,
            ..
        } if state.in_flight_requests == 1
    )));

    let atol = 1e-4f64;
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full_prefill,
        &remote_prefill,
        atol,
        atol
    )));
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full_decode,
        &remote_decode,
        atol,
        atol
    )));

    runtime.release_sequence_state_by_id(seq_id);
    let released_states = runtime.probe_stages().unwrap();
    assert!(released_states.iter().all(|state| matches!(
        state,
        RemoteStageResponse::State {
            state,
            pending_entry_replies: 0,
            ..
        } if state.in_flight_requests == 0
    )));

    stage0.shutdown().unwrap();
    stage1.shutdown().unwrap();
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote pipeline stages"]
fn pipeline_remote_runtime_llama_drain_and_shutdown_transition() {
    let model_dir = repo_model_dir("llama-3.2-1b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let (runtime, stage0, stage1) = spawn_llama_remote_runtime(&model_dir);

    let seq_id = SequenceId::from_raw(77);
    runtime.prepare_sequence_state(seq_id);
    runtime.begin_drain().unwrap();

    let draining_states = runtime.probe_stages().unwrap();
    assert!(draining_states.iter().all(|state| matches!(
        state,
        RemoteStageResponse::State { state, .. } if state.draining && !state.shutdown
    )));

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.prepare_sequence_state(SequenceId::from_raw(78));
    }));
    assert!(panic.is_err(), "draining runtime must reject new sequences");

    runtime.release_sequence_state_by_id(seq_id);
    runtime.shutdown().unwrap();

    stage0.shutdown().unwrap();
    stage1.shutdown().unwrap();
}
