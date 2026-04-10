use std::collections::HashMap;

/// Opaque identifier for one physical paged-KV block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PagedBlockId(u64);

impl PagedBlockId {
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for PagedBlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "block-{}", self.0)
    }
}

/// Static paged-KV layout shared by every sequence in one cache pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvLayout {
    pub num_layers: usize,
    pub block_size: usize,
    pub bytes_per_block: Vec<usize>,
}

impl PagedKvLayout {
    pub fn new(block_size: usize, bytes_per_block: Vec<usize>) -> Result<Self, String> {
        if block_size == 0 {
            return Err("PagedKvLayout: block_size must be > 0".to_string());
        }
        if bytes_per_block.is_empty() {
            return Err("PagedKvLayout: bytes_per_block must not be empty".to_string());
        }
        if let Some((layer_idx, bytes)) = bytes_per_block
            .iter()
            .copied()
            .enumerate()
            .find(|(_, bytes)| *bytes == 0 || *bytes % block_size != 0)
        {
            return Err(format!(
                "PagedKvLayout: layer {layer_idx} bytes_per_block ({bytes}) must be a positive multiple of block_size ({block_size})"
            ));
        }

        Ok(Self {
            num_layers: bytes_per_block.len(),
            block_size,
            bytes_per_block,
        })
    }

    pub fn uniform(
        num_layers: usize,
        block_size: usize,
        bytes_per_block: usize,
    ) -> Result<Self, String> {
        if num_layers == 0 {
            return Err("PagedKvLayout: num_layers must be > 0".to_string());
        }
        Self::new(block_size, vec![bytes_per_block; num_layers])
    }

    pub fn bytes_per_token(&self, layer_idx: usize) -> Option<usize> {
        self.bytes_per_block
            .get(layer_idx)
            .map(|bytes| bytes / self.block_size)
    }

    pub fn bytes_per_block_for_layer(&self, layer_idx: usize) -> Option<usize> {
        self.bytes_per_block.get(layer_idx).copied()
    }
}

/// Per-layer logical-to-physical mapping for paged KV storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedLayerState {
    pub block_ids: Vec<PagedBlockId>,
    pub len: usize,
    pub logical_start: usize,
}

impl PagedLayerState {
    pub fn new() -> Self {
        Self {
            block_ids: Vec::new(),
            len: 0,
            logical_start: 0,
        }
    }

    pub fn visible_len(&self) -> usize {
        self.len.saturating_sub(self.logical_start)
    }

    pub fn reserved_blocks(&self) -> usize {
        self.block_ids.len()
    }

    pub fn reserved_bytes(&self, layout: &PagedKvLayout, layer_idx: usize) -> usize {
        self.reserved_blocks()
            * layout
                .bytes_per_block_for_layer(layer_idx)
                .unwrap_or_default()
    }

    pub fn used_bytes(&self, layout: &PagedKvLayout, layer_idx: usize) -> usize {
        self.visible_len() * layout.bytes_per_token(layer_idx).unwrap_or_default()
    }
}

impl Default for PagedLayerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-sequence paged cache state spanning all transformer layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedSequenceState {
    pub block_size: usize,
    pub layers: Vec<PagedLayerState>,
}

impl PagedSequenceState {
    pub fn new(layout: &PagedKvLayout) -> Self {
        Self {
            block_size: layout.block_size,
            layers: vec![PagedLayerState::default(); layout.num_layers],
        }
    }

    pub fn layer(&self, layer_idx: usize) -> Option<&PagedLayerState> {
        self.layers.get(layer_idx)
    }

    pub fn layer_mut(&mut self, layer_idx: usize) -> Option<&mut PagedLayerState> {
        self.layers.get_mut(layer_idx)
    }

    pub fn reserved_blocks(&self) -> usize {
        self.layers
            .iter()
            .map(PagedLayerState::reserved_blocks)
            .sum()
    }

    pub fn reserved_bytes(&self, layout: &PagedKvLayout) -> usize {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| layer.reserved_bytes(layout, layer_idx))
            .sum()
    }

    pub fn used_bytes(&self, layout: &PagedKvLayout) -> usize {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| layer.used_bytes(layout, layer_idx))
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PagedBlockRecord {
    layer_idx: usize,
    in_use: bool,
}

/// Aggregated allocator/storage counters for paged KV state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagedCacheStats {
    pub allocated_blocks: usize,
    pub live_blocks: usize,
    pub free_blocks: usize,
    pub bytes_reserved: usize,
    pub bytes_in_use: usize,
}

/// Physical block allocator shared across every active paged sequence.
pub struct PagedBlockPool {
    layout: PagedKvLayout,
    next_block_id: u64,
    blocks: HashMap<PagedBlockId, PagedBlockRecord>,
    free_lists: Vec<Vec<PagedBlockId>>,
}

impl PagedBlockPool {
    pub fn new(layout: PagedKvLayout) -> Self {
        Self {
            free_lists: vec![Vec::new(); layout.num_layers],
            layout,
            next_block_id: 0,
            blocks: HashMap::new(),
        }
    }

    pub fn layout(&self) -> &PagedKvLayout {
        &self.layout
    }

    pub fn append_tokens(
        &mut self,
        state: &mut PagedSequenceState,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<(), String> {
        let num_layers = state.layers.len();
        let layer = state.layer_mut(layer_idx).ok_or_else(|| {
            format!(
                "PagedBlockPool: layer {layer_idx} out of range for {} layers",
                num_layers
            )
        })?;
        if token_count == 0 {
            return Ok(());
        }

        let new_visible_len = layer.visible_len() + token_count;
        let required_blocks = new_visible_len.div_ceil(self.layout.block_size);
        while layer.block_ids.len() < required_blocks {
            layer.block_ids.push(self.acquire_block(layer_idx)?);
        }
        layer.len += token_count;
        Ok(())
    }

    pub fn trim_tokens(
        &mut self,
        state: &mut PagedSequenceState,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<usize, String> {
        let num_layers = state.layers.len();
        let layer = state.layer_mut(layer_idx).ok_or_else(|| {
            format!(
                "PagedBlockPool: layer {layer_idx} out of range for {} layers",
                num_layers
            )
        })?;
        if token_count == 0 || layer.len == 0 {
            return Ok(0);
        }

        let min_len = layer.logical_start.min(layer.len);
        let trimmed = token_count.min(layer.len - min_len);
        if trimmed == 0 {
            return Ok(0);
        }

        layer.len -= trimmed;
        if layer.len == layer.logical_start {
            layer.logical_start = 0;
        }

        let required_blocks = layer.visible_len().div_ceil(self.layout.block_size);
        while layer.block_ids.len() > required_blocks {
            if let Some(block_id) = layer.block_ids.pop() {
                self.release_block(block_id)?;
            }
        }
        Ok(trimmed)
    }

    pub fn rewind_tokens(
        &mut self,
        state: &mut PagedSequenceState,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<usize, String> {
        self.trim_tokens(state, layer_idx, token_count)
    }

    pub fn release_sequence(&mut self, state: &mut PagedSequenceState) -> Result<(), String> {
        for layer in &mut state.layers {
            while let Some(block_id) = layer.block_ids.pop() {
                self.release_block(block_id)?;
            }
            layer.len = 0;
            layer.logical_start = 0;
        }
        Ok(())
    }

    pub fn stats_for_sequences<'a>(
        &self,
        sequences: impl IntoIterator<Item = &'a PagedSequenceState>,
    ) -> PagedCacheStats {
        let allocated_blocks = self.blocks.len();
        let free_blocks = self.blocks.values().filter(|record| !record.in_use).count();
        let live_blocks = allocated_blocks.saturating_sub(free_blocks);
        let states: Vec<&PagedSequenceState> = sequences.into_iter().collect();
        let bytes_reserved = states
            .iter()
            .map(|state| state.reserved_bytes(&self.layout))
            .sum();
        let bytes_in_use = states
            .iter()
            .map(|state| state.used_bytes(&self.layout))
            .sum();

        PagedCacheStats {
            allocated_blocks,
            live_blocks,
            free_blocks,
            bytes_reserved,
            bytes_in_use,
        }
    }

    fn acquire_block(&mut self, layer_idx: usize) -> Result<PagedBlockId, String> {
        self.validate_layer(layer_idx)?;
        if let Some(block_id) = self.free_lists[layer_idx].pop() {
            let record = self
                .blocks
                .get_mut(&block_id)
                .expect("free-list block must exist in registry");
            record.in_use = true;
            return Ok(block_id);
        }

        let block_id = PagedBlockId(self.next_block_id);
        self.next_block_id += 1;
        self.blocks.insert(
            block_id,
            PagedBlockRecord {
                layer_idx,
                in_use: true,
            },
        );
        Ok(block_id)
    }

    fn release_block(&mut self, block_id: PagedBlockId) -> Result<(), String> {
        let record = self
            .blocks
            .get_mut(&block_id)
            .ok_or_else(|| format!("PagedBlockPool: unknown block {block_id}"))?;
        if !record.in_use {
            return Err(format!(
                "PagedBlockPool: block {block_id} was already released"
            ));
        }
        record.in_use = false;
        self.free_lists[record.layer_idx].push(block_id);
        Ok(())
    }

    fn validate_layer(&self, layer_idx: usize) -> Result<(), String> {
        if layer_idx >= self.layout.num_layers {
            return Err(format!(
                "PagedBlockPool: layer {layer_idx} out of range for {} layers",
                self.layout.num_layers
            ));
        }
        Ok(())
    }
}
