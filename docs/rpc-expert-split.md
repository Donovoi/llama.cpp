# RPC Expert Split Buffer for Distributed MoE Models

This document describes the RPC Expert Split Buffer feature, which enables loading
large Mixture-of-Experts (MoE) models across multiple machines when no single machine
has enough VRAM to hold all expert tensors.

## Overview

MoE models like Kimi-K2-Thinking (360GB) have expert weight tensors (`ffn_*_exps`)
that can be extremely large. The RPC expert split buffer allows distributing these
tensors across multiple RPC servers, with each server holding a portion of the experts.

## Usage

### Starting RPC Servers

On each machine that will host a portion of the model:

```bash
./llama-rpc-server --host 0.0.0.0 --port 50052
```

### Loading the Model

Use the `--rpc-expert-split` argument to specify how to distribute expert tensors:

```bash
./llama-cli -m model.gguf \
  --rpc-expert-split "server1:50052,server2:50052,server3:50052|0.4,0.35,0.25"
```

The format is: `endpoint1,endpoint2,...|split1,split2,...`

- **endpoints**: RPC server addresses (host:port)
- **splits**: Proportional VRAM allocation (will be normalized to sum to 1.0)

### Example: Kimi-K2-Thinking on 5 Machines

Given machines with varying VRAM:
- server1: 24GB VRAM (largest share)
- server2: 12GB VRAM
- server3: 8GB VRAM
- server4: 8GB VRAM  
- server5: 6GB VRAM

```bash
./llama-cli -m Kimi-K2-Thinking-F8_Q4_K_M.gguf \
  --rpc-expert-split "server1:50052,server2:50052,server3:50052,server4:50052,server5:50052|24,12,8,8,6"
```

## How It Works

1. When the model is loaded, tensors matching the expert pattern (`ffn_(up|down|gate)_exps`)
   are allocated using the RPC split buffer type.

2. The split buffer distributes tensor **rows** across the specified endpoints according
   to the split proportions.

3. When tensor data is loaded:
   - Each endpoint receives only its portion of the rows
   - The row assignment is deterministic based on the split proportions

4. During inference (current limitation):
   - Expert tensors must be gathered to a single device for MUL_MAT_ID
   - This creates a bottleneck for very large expert tensors
   - Future work will implement distributed MUL_MAT_ID

## Limitations

1. **Compute is not yet distributed**: Currently, expert tensors are distributed for
   storage only. The actual MUL_MAT_ID computation requires gathering data to one
   device, which limits the practical benefit.

2. **Network bandwidth**: Moving expert data during inference can be slow. Best
   suited for:
   - Models that fit in aggregate VRAM but not single-device VRAM
   - Batch inference where the transfer overhead is amortized

3. **Maximum 16 endpoints**: The current implementation supports up to 16 RPC servers.

## API Reference

### C API

```c
// Create a split buffer type for distributing rows across endpoints
ggml_backend_buffer_type_t ggml_backend_rpc_split_buffer_type(
    const char ** endpoints,      // NULL-terminated array of "host:port" strings
    const uint32_t * devices,     // Device index on each endpoint (usually all 0)
    const float * tensor_split,   // Proportional split (will be normalized)
    int n_endpoints               // Number of endpoints
);

// Check if a buffer type is an RPC split buffer
bool ggml_backend_buft_is_rpc_split(ggml_backend_buffer_type_t buft);
```

### CLI Arguments

```
--rpc-expert-split CONFIG
    Distribute MoE expert tensors across RPC endpoints.
    Format: 'endpoint1:port,endpoint2:port,...|split1,split2,...'
    Example: '192.168.1.10:50052,192.168.1.11:50052|0.6,0.4'
    Splits are proportional (will be normalized to sum to 1)
```

## Future Work

- [x] Implement distributed MUL_MAT_ID to avoid gathering expert data
- [x] Add expert-based (rather than row-based) splitting for MoE tensors
- [x] Add profiling and monitoring for distributed inference
- [ ] Support dynamic load balancing based on actual VRAM usage
- [ ] Implement runtime expert migration based on activation patterns
- [ ] Add support for expert replication (hot experts on multiple endpoints)

## Implementation Status (Completed)

### Expert-Based Splitting (ggml-rpc-split.inc)

The split buffer now supports true expert-based splitting on dimension 2 (`n_expert`)
rather than row-based splitting. This keeps complete experts together on each endpoint:

```cpp
// Get expert range for a device
std::pair<int, int> rpc_split_get_expert_range(
    int device_id, int n_devices, int n_expert, 
    const std::vector<size_t>& vram);

// Get which device owns a specific expert  
int rpc_split_get_expert_endpoint(
    int expert_id, int n_devices, int n_expert,
    const std::vector<size_t>& vram);
```

### Distributed MUL_MAT_ID (ggml-rpc.cpp)

A new RPC command `RPC_CMD_MUL_MAT_ID_PARTIAL` enables executing partial MUL_MAT_ID
operations on each endpoint for only the experts they own:

1. Client queries which experts are on which endpoints
2. Client fans out partial MUL_MAT_ID requests in parallel
3. Each endpoint computes only tokens routed to its experts
4. Client accumulates partial results into final output

### Profiling Infrastructure (ggml-rpc-split.inc)

Runtime profiling tracks:
- Per-endpoint compute time (min/max/avg)
- Load balance factor across endpoints
- Expert activation frequency

```cpp
// Enable/disable profiling
void rpc_split_profile_enable();
void rpc_split_profile_disable();

// Get profiling statistics
rpc_split_profile_stats rpc_split_profile_get();
```

## Tests

The test suite (`tests/test-rpc-split.cpp`) includes 24 tests covering:
- Row-based splitting (original functionality)
- Expert-based splitting (dimension 2)
- Distributed MUL_MAT_ID routing and accumulation
- Profiling infrastructure
- Integration scenarios with small MoE models
