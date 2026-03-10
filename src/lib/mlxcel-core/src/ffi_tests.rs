// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

#[test]
fn test_zeros() {
    let arr = zeros(&[2, 3], dtype::FLOAT32);
    assert!(!arr.is_null());
    assert_eq!(array_shape(&arr), vec![2, 3]);
    assert_eq!(array_dtype(&arr), dtype::FLOAT32);
    assert_eq!(array_size(&arr), 6);
}

#[test]
fn test_ones() {
    let arr = ones(&[4, 5], dtype::FLOAT32);
    assert!(!arr.is_null());
    eval(&arr);
    let sum = sum_all(&arr);
    eval(&sum);
    assert_eq!(item_f32(&sum), 20.0);
}

#[test]
fn test_matmul() {
    let a = ones(&[2, 3], dtype::FLOAT32);
    let b = ones(&[3, 4], dtype::FLOAT32);
    let c = matmul(&a, &b);
    assert_eq!(array_shape(&c), vec![2, 4]);
    eval(&c);
    let total = sum_all(&c);
    eval(&total);
    assert_eq!(item_f32(&total), 24.0);
}

#[test]
fn test_add_multiply() {
    let a = full_f32(&[2, 2], 2.0, dtype::FLOAT32);
    let b = full_f32(&[2, 2], 3.0, dtype::FLOAT32);
    let c = add(&a, &b);
    let d = multiply(&a, &b);

    eval(&c);
    eval(&d);

    let sum = sum_all(&c);
    let prod_sum = sum_all(&d);

    eval(&sum);
    eval(&prod_sum);

    assert_eq!(item_f32(&sum), 20.0);
    assert_eq!(item_f32(&prod_sum), 24.0);
}

#[test]
fn test_softmax() {
    let a = from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let s = softmax(&a, -1);
    eval(&s);

    let total = sum_all(&s);
    eval(&total);
    let sum_val = item_f32(&total);
    assert!((sum_val - 1.0).abs() < 1e-5);
}

#[test]
fn test_rms_norm() {
    let x = ones(&[1, 4], dtype::FLOAT32);
    let weight = ones(&[4], dtype::FLOAT32);
    let normed = rms_norm(&x, &weight, 1e-5);
    eval(&normed);

    let total = sum_all(&normed);
    eval(&total);
    assert!((item_f32(&total) - 4.0).abs() < 1e-4);
}

#[test]
fn test_argmax() {
    let a = from_slice_f32(&[1.0, 3.0, 2.0, 4.0], &[1, 4]);
    let idx = argmax(&a, -1, false);
    eval(&idx);
    assert_eq!(item_i32(&idx), 3);
}

#[test]
fn test_swiglu_mlp() {
    let x = ones(&[1, 4], dtype::FLOAT32);
    let gate = ones(&[8, 4], dtype::FLOAT32);
    let up = ones(&[8, 4], dtype::FLOAT32);
    let down = ones(&[4, 8], dtype::FLOAT32);

    let out = swiglu_mlp_forward(&x, &gate, &up, &down);
    eval(&out);
    assert_eq!(array_shape(&out), vec![1, 4]);
}

#[test]
fn test_compiled_swiglu_activation() {
    let gate = from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let x = from_slice_f32(&[2.0, 2.0, 2.0, 2.0], &[1, 4]);

    let out = compiled_swiglu_activation(&gate, &x);
    eval(&out);

    assert_eq!(array_shape(&out), vec![1, 4]);

    let total = sum_all(&out);
    eval(&total);
    assert!(item_f32(&total) > 0.0);
}

#[test]
fn test_new_ops() {
    let x = from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let s = silu(&x);
    eval(&s);
    assert_eq!(array_shape(&s), vec![1, 4]);

    let g = gelu(&x);
    eval(&g);
    assert_eq!(array_shape(&g), vec![1, 4]);

    let r = relu(&x);
    eval(&r);
    assert_eq!(array_shape(&r), vec![1, 4]);

    let indices = from_slice_i32(&[0, 2], &[2]);
    let taken = take(&x, &indices, -1);
    eval(&taken);
    assert_eq!(array_shape(&taken), vec![1, 2]);

    let vals = from_slice_f32(&[3.0, 1.0, 4.0, 2.0], &[4]);
    let sorted_idx = argsort(&vals, 0);
    eval(&sorted_idx);
    assert_eq!(array_shape(&sorted_idx), vec![4]);

    let part_idx = argpartition(&vals, 1, 0);
    eval(&part_idx);
    assert_eq!(array_shape(&part_idx), vec![4]);

    let inp = ones(&[1, 4], dtype::FLOAT32);
    let weight = ones(&[4], dtype::FLOAT32);
    let normed = fast_rms_norm(&inp, &weight, 1e-5);
    eval(&normed);
    assert_eq!(array_shape(&normed), vec![1, 4]);

    let y = ones(&[2, 2], dtype::FLOAT32);
    async_eval(&y);
    synchronize_default();
    let total = sum_all(&y);
    eval(&total);
    assert_eq!(item_f32(&total), 4.0);
}

#[test]
fn test_gather_mm() {
    let a_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();

    let a = from_slice_f32(&a_data, &[2, 3, 4]);
    let b = from_slice_f32(&b_data, &[2, 4, 5]);
    let rhs_indices = from_slice_i32(&[0, 1], &[2]);

    let result = unsafe {
        gather_mm(
            &a,
            &b,
            std::ptr::null(),
            rhs_indices.as_ref().unwrap() as *const _,
            false,
        )
    };
    eval(&result);
    assert_eq!(array_shape(&result), vec![2, 3, 5]);
}

#[test]
fn test_memory_functions() {
    let max_size = gpu_max_memory_size();
    assert!(max_size > 0);

    let _old = set_wired_limit(1024 * 1024 * 1024);
    let limit = get_wired_limit();
    assert!(limit > 0);
    set_wired_limit(0);
}

#[test]
fn bench_compiled_vs_uncompiled_swiglu() {
    use std::time::Instant;

    let test_dims = [4096, 8192, 14336, 24576, 49152];

    for dim in test_dims {
        let gate_data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();
        let x_data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002).cos()).collect();

        let gate = from_slice_f32(&gate_data, &[1, dim]);
        let x = from_slice_f32(&x_data, &[1, dim]);

        for _ in 0..10 {
            let out = compiled_swiglu_activation(&gate, &x);
            eval(&out);
        }

        let iterations = 200;
        let start = Instant::now();
        for _ in 0..iterations {
            let out = compiled_swiglu_activation(&gate, &x);
            eval(&out);
        }
        let compiled_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            let silu_gate = multiply(&gate, &sigmoid(&gate));
            let out = multiply(&silu_gate, &x);
            eval(&out);
        }
        let uncompiled_time = start.elapsed();

        println!(
            "dim={:5} | Compiled: {:.2} μs | Uncompiled: {:.2} μs | Speedup: {:.2}x",
            dim,
            compiled_time.as_micros() as f64 / iterations as f64,
            uncompiled_time.as_micros() as f64 / iterations as f64,
            uncompiled_time.as_secs_f64() / compiled_time.as_secs_f64()
        );
    }
}
