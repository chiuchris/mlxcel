use super::require_array_ref;
use mlxcel_core::{self, UniquePtr, dtype};

#[test]
fn require_array_ref_rejects_null_unique_ptrs() {
    let array: UniquePtr<mlxcel_core::MlxArray> = UniquePtr::null();
    let err = match require_array_ref(&array, "test array") {
        Ok(_) => panic!("expected null pointer to fail"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("null test array"));
}

#[test]
fn require_array_ref_accepts_real_arrays() {
    let array = mlxcel_core::ones(&[1, 2], dtype::FLOAT32);
    let resolved = require_array_ref(&array, "test array").unwrap();
    assert_eq!(mlxcel_core::array_shape(resolved), vec![1, 2]);
}
