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

//! Stream-selection wrappers for generation-time pipelining.

use crate::ffi;
use crate::ffi::MlxStream;
use crate::UniquePtr;

/// Used by: CxxGenerator, SpeculativeGenerator
pub(crate) fn new_generation_stream() -> Option<UniquePtr<MlxStream>> {
    if ffi::is_gpu_available() {
        Some(ffi::new_gpu_stream())
    } else {
        None
    }
}

/// Used by: CxxGenerator, SpeculativeGenerator
pub(crate) fn install_default_stream(stream: Option<&UniquePtr<MlxStream>>) {
    if let Some(stream) = stream {
        ffi::set_default_stream(stream);
    }
}
