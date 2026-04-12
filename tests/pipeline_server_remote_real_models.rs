mod common;

use std::net::TcpListener;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::repo_model_dir;
use mlxcel::SamplingConfig;
use mlxcel::distributed::TransportBackend;
use mlxcel::distributed::pipeline::{
    RemotePipelineRuntimeConfig, RemoteStageServiceConfig, RemoteStageServiceHandle,
    resolve_in_process_pipeline_num_layers, resolve_in_process_stage_assignments,
};
use mlxcel::server::batch::{BatchObservability, RequestPriority};
use mlxcel::server::{
    BatchMetrics, ModelProvider, PipelineParallelRuntimeConfig, ServerConfig, ServerGenerateOptions,
};

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

fn wait_for_loaded(provider: &ModelProvider) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if provider.is_loaded() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("model provider did not finish loading within 30s");
}

#[test]
#[ignore = "requires local model weights and TCP-bound remote stage services"]
fn pipeline_server_remote_coordinator_llama_matches_dense_baseline() {
    let model_dir = repo_model_dir("llama-3.2-1b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let num_layers = resolve_in_process_pipeline_num_layers(&model_dir).unwrap();
    let assignments =
        resolve_in_process_stage_assignments(num_layers, None, Some("0-7,8-15")).unwrap();

    let coordinator_addr = format!("127.0.0.1:{}", reserve_port());
    let stage0_addr = format!("127.0.0.1:{}", reserve_port());
    let stage1_addr = format!("127.0.0.1:{}", reserve_port());

    let stage1 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.clone(),
        bind_address: stage1_addr.clone(),
        stage_assignment: assignments[1].clone(),
        num_stages: 2,
        upstream_peer: Some(stage0_addr.clone()),
        downstream_peer: None,
    })
    .unwrap();
    let stage0 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.clone(),
        bind_address: stage0_addr.clone(),
        stage_assignment: assignments[0].clone(),
        num_stages: 2,
        upstream_peer: None,
        downstream_peer: Some(stage1_addr.clone()),
    })
    .unwrap();

    let dense_provider = ModelProvider::new_with_server_config(
        model_dir.clone(),
        None,
        &ServerConfig::default(),
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
    )
    .unwrap();
    let remote_provider = ModelProvider::new_with_server_config(
        model_dir.clone(),
        None,
        &ServerConfig {
            pipeline_parallel_runtime: Some(PipelineParallelRuntimeConfig::RemoteCoordinator(
                RemotePipelineRuntimeConfig {
                    stage_peers: vec![stage0_addr.clone(), stage1_addr.clone()],
                    transport_backend: TransportBackend::Tcp,
                    bind_address: coordinator_addr,
                },
            )),
            ..ServerConfig::default()
        },
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
    )
    .unwrap();

    wait_for_loaded(&dense_provider);
    wait_for_loaded(&remote_provider);

    let options = ServerGenerateOptions {
        max_tokens: 8,
        sampling: SamplingConfig::greedy(),
        stop_sequences: None,
        priority: RequestPriority::Normal,
        logprobs: Default::default(),
    };

    let dense = dense_provider
        .generate("Hello".to_string(), options.clone())
        .unwrap();
    let remote = remote_provider
        .generate("Hello".to_string(), options)
        .unwrap();

    drop(remote_provider);
    drop(dense_provider);
    stage0.shutdown().unwrap();
    stage1.shutdown().unwrap();

    assert_eq!(dense.text, remote.text);
    assert_eq!(dense.completion_tokens, remote.completion_tokens);
}
