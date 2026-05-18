fn main() {
    use mlxcel::models::cxx::llama3::Llama3Model;
    use mlxcel_core::layers::KVCache;
    use std::time::Instant;

    let (model, _) = Llama3Model::load("models/Meta-Llama-3.1-8B-Instruct-4bit").unwrap();
    let input = mlxcel_core::from_slice_i32(&[9906], &[1, 1]);

    // Warmup
    println!("Warming up...");
    for _ in 0..20 {
        let mut caches: Vec<KVCache> = model.make_caches();
        let logits = model.forward(&input, &mut caches, None);
        mlxcel_core::eval(&logits);
    }
    mlxcel_core::synchronize_default();

    // Benchmark
    println!("Benchmarking fresh cache...");
    let mut times = Vec::new();
    for _ in 0..50 {
        let mut caches: Vec<KVCache> = model.make_caches();
        mlxcel_core::synchronize_default(); // Sync before timing

        let start = Instant::now();
        let logits = model.forward(&input, &mut caches, None);
        mlxcel_core::eval(&logits);
        mlxcel_core::synchronize_default(); // Sync after
        times.push(start.elapsed());
    }

    let avg = times.iter().map(|t| t.as_secs_f64() * 1000.0).sum::<f64>() / times.len() as f64;
    let min = times
        .iter()
        .map(|t| t.as_secs_f64() * 1000.0)
        .fold(f64::INFINITY, f64::min);

    println!("Fresh cache avg: {:.2} ms", avg);
    println!("Fresh cache min: {:.2} ms", min);
    println!("Implied tok/s: {:.2}", 1000.0 / avg);
}
