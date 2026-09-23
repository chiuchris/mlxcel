use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use memmap2::Mmap;
use mlxcel::initialize_runtime;
use mlxcel_core::drafter::dflash::DFlashDrafter;
use mlxcel_core::layers::{KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{dtype, from_bytes, from_bytes_f16};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const DRAFT_BLOCK_SIZE: usize = 8;

#[derive(Parser)]
#[command(about = "Persistent DFlash draft sidecar over JSONL")]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    bindings: PathBuf,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Reset,
    Draft {
        last_bonus: i32,
        capture_layer_ids: Vec<usize>,
        capture_rows: usize,
        hidden_size: usize,
        hidden_f16: Vec<u16>,
    },
}

#[derive(Serialize)]
struct Response {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    proposals: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

struct Sidecar {
    drafter: DFlashDrafter,
    cache: Vec<KVCache>,
}

impl Sidecar {
    fn load(checkpoint: &std::path::Path, bindings: &std::path::Path) -> Result<Self> {
        let mut drafter = DFlashDrafter::load(checkpoint)
            .map_err(|error| anyhow::anyhow!("load DFlash checkpoint: {error}"))?;
        if !drafter.model.needs_embed_binding() || !drafter.model.needs_lm_head_binding() {
            bail!("checkpoint must omit target embedding and untied LM head");
        }

        let (embedding, lm_head) = load_target_bindings(bindings, &drafter)?;
        drafter.model.bind_target_embedding(embedding);
        drafter.model.bind_target_lm_head(Some(lm_head));
        let cache = drafter.model.make_cache();
        Ok(Self { drafter, cache })
    }

    fn handle(&mut self, request: Request) -> Result<Response> {
        match request {
            Request::Reset => {
                self.cache = self.drafter.model.make_cache();
                Ok(Response {
                    ok: true,
                    proposals: None,
                    error: None,
                })
            }
            Request::Draft {
                last_bonus,
                capture_layer_ids,
                capture_rows,
                hidden_size,
                hidden_f16,
            } => {
                let config = &self.drafter.model.config;
                validate_capture(
                    capture_rows,
                    &capture_layer_ids,
                    hidden_size,
                    hidden_f16.len(),
                    &config.target_layer_ids,
                    config.hidden_size,
                )?;
                if !(0..config.vocab_size as i32).contains(&last_bonus) {
                    bail!("last_bonus token {last_bonus} is outside the DFlash vocabulary");
                }

                let combined_hidden = capture_layer_ids
                    .len()
                    .checked_mul(hidden_size)
                    .context("capture width overflow")?;
                let rows =
                    i32::try_from(capture_rows).context("capture row count exceeds MLX limits")?;
                let width =
                    i32::try_from(combined_hidden).context("capture width exceeds MLX limits")?;
                let hidden_bytes: Vec<u8> = hidden_f16
                    .iter()
                    .flat_map(|bits| bits.to_le_bytes())
                    .collect();
                let target_hidden = from_bytes_f16(&hidden_bytes, &[1, rows, width], false);
                if target_hidden.is_null() {
                    bail!("MLX could not create the captured target-hidden tensor");
                }

                let proposals = self.drafter.model.draft_block(
                    last_bonus,
                    target_hidden.as_ref().expect("nonnull array checked above"),
                    &mut self.cache,
                    DRAFT_BLOCK_SIZE,
                );
                if proposals.len() != DRAFT_BLOCK_SIZE - 1 {
                    bail!("DFlash returned {} proposals; expected 7", proposals.len());
                }
                Ok(Response {
                    ok: true,
                    proposals: Some(proposals),
                    error: None,
                })
            }
        }
    }
}

fn load_target_bindings(
    path: &std::path::Path,
    drafter: &DFlashDrafter,
) -> Result<(UnifiedEmbedding, UnifiedLinear)> {
    let file = File::open(path).with_context(|| format!("open bindings {}", path.display()))?;
    // Safety: this read-only mapping stays alive until all tensor arrays have
    // been copied into MLX-owned buffers below.
    let mapped = unsafe { Mmap::map(&file) }.context("map target binding file")?;
    let tensors = SafeTensors::deserialize(&mapped).context("parse target binding safetensors")?;
    let header = read_header(&mapped)?;
    let metadata = header
        .get("__metadata__")
        .and_then(Value::as_object)
        .context("binding file has no safetensors metadata")?;

    let config = &drafter.model.config;
    let expected_shape = (config.vocab_size, config.hidden_size);
    let mut weights = WeightMap::new();
    load_quantized_binding(
        &tensors,
        metadata,
        "target.embed_tokens",
        "embed_tokens",
        expected_shape,
        &mut weights,
    )?;
    load_quantized_binding(
        &tensors,
        metadata,
        "target.lm_head",
        "lm_head",
        expected_shape,
        &mut weights,
    )?;

    let embedding = UnifiedEmbedding::from_weights(&weights, "embed_tokens", 64, 4)
        .map_err(|error| anyhow::anyhow!("load target embedding binding: {error}"))?;
    let lm_head = UnifiedLinear::from_weights(&weights, "lm_head", 64, 4)
        .map_err(|error| anyhow::anyhow!("load target LM-head binding: {error}"))?;
    Ok((embedding, lm_head))
}

fn load_quantized_binding(
    tensors: &SafeTensors<'_>,
    metadata: &serde_json::Map<String, Value>,
    source_prefix: &str,
    weight_prefix: &str,
    expected_shape: (usize, usize),
    weights: &mut WeightMap,
) -> Result<()> {
    let shape_key = format!("{source_prefix}.logical_shape");
    let shape = metadata
        .get(&shape_key)
        .and_then(Value::as_str)
        .context("binding logical shape is missing")?;
    let dimensions = parse_shape(shape)?;
    if dimensions != expected_shape {
        bail!("{source_prefix} shape {dimensions:?} does not match {expected_shape:?}");
    }
    for (field, expected) in [
        ("quantization_bits", "4"),
        ("quantization_group_size", "64"),
    ] {
        let key = format!("{source_prefix}.{field}");
        if metadata.get(&key).and_then(Value::as_str) != Some(expected) {
            bail!("{source_prefix} has incompatible {field}");
        }
    }
    if metadata
        .get(&format!("{source_prefix}.packing"))
        .and_then(Value::as_str)
        != Some("gturbo_q4_affine")
    {
        bail!("{source_prefix} does not use the supported GTurbo Q4 affine packing");
    }

    let (vocab_size, hidden_size) = dimensions;
    if hidden_size % 64 != 0 || hidden_size % 8 != 0 {
        bail!("{source_prefix} dimensions are incompatible with Q4 group size 64");
    }
    let packed = tensors
        .tensor(&format!("{source_prefix}.weight"))
        .with_context(|| format!("missing {source_prefix}.weight"))?;
    let scales = tensors
        .tensor(&format!("{source_prefix}.scales"))
        .with_context(|| format!("missing {source_prefix}.scales"))?;
    let biases = tensors
        .tensor(&format!("{source_prefix}.biases"))
        .with_context(|| format!("missing {source_prefix}.biases"))?;
    let expected_packed_bytes = vocab_size
        .checked_mul(hidden_size)
        .context("packed tensor byte count overflow")?
        / 2;
    let expected_aux_bytes = vocab_size
        .checked_mul(hidden_size / 64)
        .and_then(|count| count.checked_mul(2))
        .context("quantization auxiliary byte count overflow")?;
    if packed.dtype() != Dtype::U8 || packed.data().len() != expected_packed_bytes {
        bail!("{source_prefix}.weight has an incompatible packed dtype or length");
    }
    if scales.dtype() != Dtype::BF16 || scales.data().len() != expected_aux_bytes {
        bail!("{source_prefix}.scales has an incompatible dtype or length");
    }
    if biases.dtype() != Dtype::BF16 || biases.data().len() != expected_aux_bytes {
        bail!("{source_prefix}.biases has an incompatible dtype or length");
    }

    let packed_shape = [
        i32::try_from(vocab_size).context("vocabulary exceeds MLX limits")?,
        i32::try_from(hidden_size / 8).context("packed hidden width exceeds MLX limits")?,
    ];
    let auxiliary_shape = [
        i32::try_from(vocab_size).context("vocabulary exceeds MLX limits")?,
        i32::try_from(hidden_size / 64).context("quantization group count exceeds MLX limits")?,
    ];
    let weight = from_bytes(packed.data(), &packed_shape, dtype::UINT32);
    let scale = from_bytes_f16(scales.data(), &auxiliary_shape, true);
    let bias = from_bytes_f16(biases.data(), &auxiliary_shape, true);
    if weight.is_null() || scale.is_null() || bias.is_null() {
        bail!("MLX could not create {source_prefix} quantized arrays");
    }
    weights.insert(format!("{weight_prefix}.weight"), weight);
    weights.insert(format!("{weight_prefix}.scales"), scale);
    weights.insert(format!("{weight_prefix}.biases"), bias);
    Ok(())
}

fn read_header(bytes: &[u8]) -> Result<Value> {
    if bytes.len() < 8 {
        bail!("binding file is shorter than a safetensors header prefix");
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))
        .context("safetensors header length exceeds host address space")?;
    let header_end = 8_usize
        .checked_add(header_len)
        .context("safetensors header length overflow")?;
    let header = bytes
        .get(8..header_end)
        .context("safetensors header is truncated")?;
    serde_json::from_slice(header).context("parse binding metadata header")
}

fn parse_shape(value: &str) -> Result<(usize, usize)> {
    let mut dimensions = value.split(',').map(str::parse::<usize>);
    let vocab_size = dimensions
        .next()
        .context("missing vocabulary dimension")??;
    let hidden_size = dimensions.next().context("missing hidden dimension")??;
    if dimensions.next().is_some() {
        bail!("binding logical shape must contain exactly two dimensions");
    }
    Ok((vocab_size, hidden_size))
}

fn validate_capture(
    rows: usize,
    layer_ids: &[usize],
    hidden_size: usize,
    value_count: usize,
    expected_layer_ids: &[usize],
    expected_hidden_size: usize,
) -> Result<()> {
    if rows == 0 {
        bail!("target hidden capture must contain at least one row");
    }
    if layer_ids != expected_layer_ids {
        bail!("target hidden capture layer IDs do not match the DFlash checkpoint");
    }
    if hidden_size != expected_hidden_size {
        bail!("target hidden size {hidden_size} does not match {expected_hidden_size}");
    }
    let expected_count = rows
        .checked_mul(layer_ids.len())
        .and_then(|count| count.checked_mul(hidden_size))
        .context("target hidden capture size overflow")?;
    if value_count != expected_count {
        bail!("target hidden capture has {value_count} values; expected {expected_count}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _runtime = initialize_runtime();
    let mut sidecar = Sidecar::load(&args.checkpoint, &args.bindings)?;
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());

    for line in stdin.lock().lines() {
        let line = line.context("read JSONL request")?;
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(request) => match sidecar.handle(request) {
                Ok(response) => response,
                Err(error) => Response {
                    ok: false,
                    proposals: None,
                    error: Some(format!("{error:#}")),
                },
            },
            Err(error) => Response {
                ok: false,
                proposals: None,
                error: Some(format!("invalid request: {error}")),
            },
        };
        serde_json::to_writer(&mut stdout, &response).context("write JSONL response")?;
        stdout
            .write_all(b"\n")
            .context("terminate JSONL response")?;
        stdout.flush().context("flush JSONL response")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_shape, validate_capture};

    #[test]
    fn accepts_exact_qwen36_capture_contract() {
        let layer_ids = [1, 6, 11, 16, 22, 27, 32, 37];
        assert!(validate_capture(3, &layer_ids, 2048, 3 * 8 * 2048, &layer_ids, 2048).is_ok());
    }

    #[test]
    fn rejects_capture_with_wrong_layer_order_or_length() {
        let expected = [1, 6, 11, 16, 22, 27, 32, 37];
        let wrong = [1, 6, 11, 16, 22, 27, 32, 36];
        assert!(validate_capture(1, &wrong, 2048, 8 * 2048, &expected, 2048).is_err());
        assert!(validate_capture(1, &expected, 2048, 7 * 2048, &expected, 2048).is_err());
    }

    #[test]
    fn parses_only_two_dimensional_binding_shapes() {
        assert_eq!(parse_shape("248320,2048").unwrap(), (248320, 2048));
        assert!(parse_shape("248320,2048,1").is_err());
    }
}
