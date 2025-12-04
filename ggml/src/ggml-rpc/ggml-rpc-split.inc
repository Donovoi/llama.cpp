// RPC Split Buffer Implementation
// This file contains the implementation of ggml_backend_rpc_split_buffer_type
// which distributes tensor rows across multiple RPC endpoints for MoE models.
//
// Usage:
//   const char * endpoints[] = {"host1:50052", "host2:50052", "host3:50052", nullptr};
//   uint32_t devices[] = {0, 0, 0};
//   float tensor_split[] = {0.4f, 0.35f, 0.25f};
//   auto buft = ggml_backend_rpc_split_buffer_type(endpoints, devices, tensor_split, 3);
//
// This is designed to be #included into ggml-rpc.cpp after the regular buffer type implementation.

#include <array>
#include <algorithm>

// Maximum number of RPC endpoints in a split buffer
#define GGML_RPC_SPLIT_MAX_ENDPOINTS 16

// Context for split buffer type - stores endpoint info and split proportions
struct ggml_backend_rpc_split_buffer_type_context {
    std::vector<std::string> endpoints;
    std::vector<uint32_t> devices;
    std::array<float, GGML_RPC_SPLIT_MAX_ENDPOINTS> tensor_split;
    int n_endpoints;
    std::string name;
};

// Per-tensor data stored in tensor->extra for split tensors
struct ggml_backend_rpc_split_tensor_extra {
    // Remote buffer pointers on each endpoint
    std::array<uint64_t, GGML_RPC_SPLIT_MAX_ENDPOINTS> remote_ptrs;
    // Socket connections for each endpoint
    std::array<std::shared_ptr<socket_t>, GGML_RPC_SPLIT_MAX_ENDPOINTS> sockets;
    // Row ranges for each endpoint (used for row-based split)
    std::array<int64_t, GGML_RPC_SPLIT_MAX_ENDPOINTS> row_low;
    std::array<int64_t, GGML_RPC_SPLIT_MAX_ENDPOINTS> row_high;
    // Expert ranges for each endpoint (used for expert-based split)
    std::array<int64_t, GGML_RPC_SPLIT_MAX_ENDPOINTS> expert_low;
    std::array<int64_t, GGML_RPC_SPLIT_MAX_ENDPOINTS> expert_high;
    int n_endpoints;
    bool is_expert_split;  // true = split on dim 2 (experts), false = split on rows
};

// Context for split buffer instance
struct ggml_backend_rpc_split_buffer_context {
    ~ggml_backend_rpc_split_buffer_context() {
        for (auto * extra : tensor_extras) {
            delete extra;
        }
    }
    std::vector<ggml_backend_rpc_split_tensor_extra *> tensor_extras;
    // Cached socket connections
    std::vector<std::shared_ptr<socket_t>> sockets;
    std::vector<std::string> endpoints;
    std::vector<uint32_t> devices;
    std::array<float, GGML_RPC_SPLIT_MAX_ENDPOINTS> tensor_split;
    int n_endpoints;
};

// Calculate row split for a given endpoint
static void rpc_split_get_row_split(int64_t * row_low, int64_t * row_high, 
                                    int64_t nrows, const float * tensor_split, 
                                    int n_endpoints, int endpoint_id) {
    // Calculate cumulative split proportions
    float sum = 0.0f;
    for (int i = 0; i < n_endpoints; i++) {
        sum += tensor_split[i];
    }
    if (sum == 0.0f) {
        // Default to equal split
        sum = (float)n_endpoints;
        for (int i = 0; i < n_endpoints; i++) {
            // Use 1.0f for each since we're recalculating
        }
    }
    
    float cumulative = 0.0f;
    for (int i = 0; i < endpoint_id; i++) {
        cumulative += tensor_split[i] / sum;
    }
    
    *row_low = (int64_t)(nrows * cumulative);
    
    if (endpoint_id == n_endpoints - 1) {
        *row_high = nrows;
    } else {
        cumulative += tensor_split[endpoint_id] / sum;
        *row_high = (int64_t)(nrows * cumulative);
    }
}

// Forward declarations for expert-based splitting (defined at end of file)
static void rpc_split_get_expert_range(int64_t * expert_low, int64_t * expert_high,
                                       int64_t n_expert, const float * tensor_split,
                                       int n_endpoints, int endpoint_id);
static bool rpc_split_is_expert_tensor(const ggml_tensor * tensor);

static void ggml_backend_rpc_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_split_buffer_context * ctx = (ggml_backend_rpc_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_rpc_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // Split buffer doesn't have a single base address - pointers are in tensor extras
    // Return a dummy non-null address
    GGML_UNUSED(buffer);
    return (void *)0x1000;
}

static enum ggml_status ggml_backend_rpc_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src == nullptr && "views of split tensors are not supported");
    
    ggml_backend_rpc_split_buffer_context * ctx = (ggml_backend_rpc_split_buffer_context *)buffer->context;
    
    ggml_backend_rpc_split_tensor_extra * extra = new ggml_backend_rpc_split_tensor_extra{};
    ctx->tensor_extras.push_back(extra);
    extra->n_endpoints = ctx->n_endpoints;
    
    // Determine if this is an expert tensor that should use expert-based splitting
    const bool use_expert_split = rpc_split_is_expert_tensor(tensor);
    extra->is_expert_split = use_expert_split;
    
    const int64_t n_expert = tensor->ne[2];  // For expert tensors: [embd, ff, n_expert]
    const int64_t nrows = ggml_nrows(tensor);
    
    LOG_DBG("[%s] %s: expert_split=%d, ne=[%ld,%ld,%ld,%ld]\n", 
            __func__, tensor->name, use_expert_split, 
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    
    // Allocate memory on each endpoint for its portion of the tensor
    for (int i = 0; i < ctx->n_endpoints; i++) {
        int64_t split_low, split_high;
        int64_t split_dim_size;
        size_t size;
        
        if (use_expert_split) {
            // Expert-based split: each endpoint gets complete experts
            rpc_split_get_expert_range(&split_low, &split_high, n_expert, 
                                       ctx->tensor_split.data(), ctx->n_endpoints, i);
            extra->expert_low[i] = split_low;
            extra->expert_high[i] = split_high;
            extra->row_low[i] = 0;
            extra->row_high[i] = 0;
            
            split_dim_size = split_high - split_low;
            if (split_dim_size == 0) {
                extra->remote_ptrs[i] = 0;
                continue;
            }
            
            // Size = elements per expert * number of experts * element size
            size_t elements_per_expert = tensor->ne[0] * tensor->ne[1];
            size = split_dim_size * elements_per_expert * ggml_type_size(tensor->type) / ggml_blck_size(tensor->type);
            
            LOG_DBG("[%s] endpoint %d: experts %ld-%ld (%ld experts), size=%zu bytes\n", 
                    __func__, i, split_low, split_high - 1, split_dim_size, size);
        } else {
            // Row-based split: distribute rows across endpoints
            rpc_split_get_row_split(&split_low, &split_high, nrows, 
                                    ctx->tensor_split.data(), ctx->n_endpoints, i);
            extra->row_low[i] = split_low;
            extra->row_high[i] = split_high;
            extra->expert_low[i] = 0;
            extra->expert_high[i] = 0;
            
            split_dim_size = split_high - split_low;
            if (split_dim_size == 0) {
                extra->remote_ptrs[i] = 0;
                continue;
            }
            
            // Size = rows * row_size
            size = split_dim_size * ggml_row_size(tensor->type, tensor->ne[0]);
            
            LOG_DBG("[%s] endpoint %d: rows %ld-%ld (%ld rows), size=%zu bytes\n", 
                    __func__, i, split_low, split_high - 1, split_dim_size, size);
        }
        
        // Get socket for this endpoint
        auto sock = ctx->sockets[i];
        if (sock == nullptr) {
            GGML_LOG_ERROR("RPC split buffer: no connection to endpoint %d\n", i);
            return GGML_STATUS_FAILED;
        }
        extra->sockets[i] = sock;
        
        // Allocate buffer on remote endpoint
        rpc_msg_alloc_buffer_req request = {ctx->devices[i], size};
        rpc_msg_alloc_buffer_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
        if (!status || response.remote_ptr == 0) {
            GGML_LOG_ERROR("RPC split buffer: failed to allocate %zu bytes on endpoint %d\n", size, i);
            return GGML_STATUS_FAILED;
        }
        
        extra->remote_ptrs[i] = response.remote_ptr;
    }
    
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_split_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, 
                                                     const void * data, size_t offset, size_t size) {
    // For split tensors, we need to distribute the data across endpoints
    // This expects the full tensor data, starting at offset
    GGML_ASSERT(offset == 0 && "split tensors must be set from offset 0");
    GGML_ASSERT(size == ggml_nbytes(tensor) && "split tensors must be set in their entirety");
    
    ggml_backend_rpc_split_tensor_extra * extra = (ggml_backend_rpc_split_tensor_extra *)tensor->extra;
    GGML_ASSERT(extra != nullptr);
    
    const uint8_t * src = (const uint8_t *)data;
    
    if (extra->is_expert_split) {
        // Expert-based split: send complete experts to each endpoint
        // Tensor shape is [ne0, ne1, n_expert] = [embd, ff, n_expert]
        const size_t expert_size = tensor->ne[0] * tensor->ne[1] * ggml_type_size(tensor->type) / ggml_blck_size(tensor->type);
        
        for (int i = 0; i < extra->n_endpoints; i++) {
            if (extra->remote_ptrs[i] == 0) {
                continue; // No experts on this endpoint
            }
            
            int64_t expert_low = extra->expert_low[i];
            int64_t expert_high = extra->expert_high[i];
            int64_t n_experts_split = expert_high - expert_low;
            
            size_t split_offset = expert_low * expert_size;
            size_t split_size = n_experts_split * expert_size;
            
            // Create tensor descriptor for this endpoint's experts
            rpc_tensor rpc_tensor_desc;
            memset(&rpc_tensor_desc, 0, sizeof(rpc_tensor_desc));
            rpc_tensor_desc.id = (uint64_t)tensor;
            rpc_tensor_desc.type = tensor->type;
            rpc_tensor_desc.buffer = extra->remote_ptrs[i];
            rpc_tensor_desc.ne[0] = tensor->ne[0];
            rpc_tensor_desc.ne[1] = tensor->ne[1];
            rpc_tensor_desc.ne[2] = n_experts_split;  // Only this endpoint's experts
            rpc_tensor_desc.ne[3] = 1;
            rpc_tensor_desc.nb[0] = ggml_type_size(tensor->type);
            rpc_tensor_desc.nb[1] = rpc_tensor_desc.nb[0] * tensor->ne[0] / ggml_blck_size(tensor->type);
            rpc_tensor_desc.nb[2] = rpc_tensor_desc.nb[1] * tensor->ne[1];
            rpc_tensor_desc.nb[3] = rpc_tensor_desc.nb[2] * n_experts_split;
            rpc_tensor_desc.data = extra->remote_ptrs[i];
            snprintf(rpc_tensor_desc.name, GGML_MAX_NAME, "%.50s_e%d", tensor->name, i);
            
            // Send the experts to this endpoint
            size_t input_size = sizeof(rpc_tensor_desc) + sizeof(uint64_t) + split_size;
            std::vector<uint8_t> input(input_size, 0);
            uint64_t wire_offset = 0;
            memcpy(input.data(), &rpc_tensor_desc, sizeof(rpc_tensor_desc));
            memcpy(input.data() + sizeof(rpc_tensor_desc), &wire_offset, sizeof(wire_offset));
            memcpy(input.data() + sizeof(rpc_tensor_desc) + sizeof(wire_offset), src + split_offset, split_size);
            
            bool status = send_rpc_cmd(extra->sockets[i], RPC_CMD_SET_TENSOR, input.data(), input.size());
            RPC_STATUS_ASSERT(status);
            
            LOG_DBG("[%s] endpoint %d: set %zu bytes for experts %ld-%ld\n", 
                    __func__, i, split_size, expert_low, expert_high - 1);
        }
    } else {
        // Row-based split: original implementation
        const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        
        for (int i = 0; i < extra->n_endpoints; i++) {
            if (extra->remote_ptrs[i] == 0) {
                continue; // No rows on this endpoint
            }
            
            int64_t row_low = extra->row_low[i];
            int64_t row_high = extra->row_high[i];
            int64_t nrows_split = row_high - row_low;
            
            size_t split_offset = row_low * row_size;
            size_t split_size = nrows_split * row_size;
            
            // Create a temporary tensor descriptor for this slice
            rpc_tensor rpc_tensor_desc;
            memset(&rpc_tensor_desc, 0, sizeof(rpc_tensor_desc));
            rpc_tensor_desc.id = (uint64_t)tensor;
            rpc_tensor_desc.type = tensor->type;
            rpc_tensor_desc.buffer = extra->remote_ptrs[i];
            rpc_tensor_desc.ne[0] = tensor->ne[0];
            rpc_tensor_desc.ne[1] = nrows_split;
            for (int d = 2; d < GGML_MAX_DIMS; d++) {
                rpc_tensor_desc.ne[d] = 1;
            }
            rpc_tensor_desc.nb[0] = ggml_type_size(tensor->type);
            rpc_tensor_desc.nb[1] = row_size;
            rpc_tensor_desc.data = extra->remote_ptrs[i];
            snprintf(rpc_tensor_desc.name, GGML_MAX_NAME, "%.50s_s%d", tensor->name, i);
            
            // Send the data slice to this endpoint
            size_t input_size = sizeof(rpc_tensor_desc) + sizeof(uint64_t) + split_size;
            std::vector<uint8_t> input(input_size, 0);
            uint64_t wire_offset = 0;
            memcpy(input.data(), &rpc_tensor_desc, sizeof(rpc_tensor_desc));
            memcpy(input.data() + sizeof(rpc_tensor_desc), &wire_offset, sizeof(wire_offset));
            memcpy(input.data() + sizeof(rpc_tensor_desc) + sizeof(wire_offset), src + split_offset, split_size);
            
            bool status = send_rpc_cmd(extra->sockets[i], RPC_CMD_SET_TENSOR, input.data(), input.size());
            RPC_STATUS_ASSERT(status);
            
            LOG_DBG("[%s] endpoint %d: set %zu bytes for rows %ld-%ld\n", 
                    __func__, i, split_size, row_low, row_high);
        }
    }
    
    GGML_UNUSED(buffer);
}

static void ggml_backend_rpc_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, 
                                                     void * data, size_t offset, size_t size) {
    // Gather data from all endpoints back into the full tensor
    GGML_ASSERT(offset == 0 && "split tensors must be read from offset 0");
    GGML_ASSERT(size == ggml_nbytes(tensor) && "split tensors must be read in their entirety");
    
    ggml_backend_rpc_split_tensor_extra * extra = (ggml_backend_rpc_split_tensor_extra *)tensor->extra;
    GGML_ASSERT(extra != nullptr);
    
    uint8_t * dst = (uint8_t *)data;
    
    if (extra->is_expert_split) {
        // Expert-based split: gather complete experts from each endpoint
        const size_t expert_size = tensor->ne[0] * tensor->ne[1] * ggml_type_size(tensor->type) / ggml_blck_size(tensor->type);
        
        for (int i = 0; i < extra->n_endpoints; i++) {
            if (extra->remote_ptrs[i] == 0) {
                continue;
            }
            
            int64_t expert_low = extra->expert_low[i];
            int64_t expert_high = extra->expert_high[i];
            int64_t n_experts_split = expert_high - expert_low;
            
            size_t split_offset = expert_low * expert_size;
            size_t split_size = n_experts_split * expert_size;
            
            // Create tensor descriptor for request
            rpc_tensor rpc_tensor_desc;
            memset(&rpc_tensor_desc, 0, sizeof(rpc_tensor_desc));
            rpc_tensor_desc.id = (uint64_t)tensor;
            rpc_tensor_desc.type = tensor->type;
            rpc_tensor_desc.buffer = extra->remote_ptrs[i];
            rpc_tensor_desc.ne[0] = tensor->ne[0];
            rpc_tensor_desc.ne[1] = tensor->ne[1];
            rpc_tensor_desc.ne[2] = n_experts_split;
            rpc_tensor_desc.ne[3] = 1;
            rpc_tensor_desc.data = extra->remote_ptrs[i];
            
            rpc_msg_get_tensor_req request;
            request.tensor = rpc_tensor_desc;
            request.offset = 0;
            request.size = split_size;
            
            bool status = send_rpc_cmd(extra->sockets[i], RPC_CMD_GET_TENSOR, &request, sizeof(request), 
                                       dst + split_offset, split_size);
            RPC_STATUS_ASSERT(status);
        }
    } else {
        // Row-based split: original implementation
        const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        
        for (int i = 0; i < extra->n_endpoints; i++) {
            if (extra->remote_ptrs[i] == 0) {
                continue;
            }
            
            int64_t row_low = extra->row_low[i];
            int64_t row_high = extra->row_high[i];
            int64_t nrows_split = row_high - row_low;
            
            size_t split_offset = row_low * row_size;
            size_t split_size = nrows_split * row_size;
            
            // Create tensor descriptor for request
            rpc_tensor rpc_tensor_desc;
            memset(&rpc_tensor_desc, 0, sizeof(rpc_tensor_desc));
            rpc_tensor_desc.id = (uint64_t)tensor;
            rpc_tensor_desc.type = tensor->type;
            rpc_tensor_desc.buffer = extra->remote_ptrs[i];
            rpc_tensor_desc.ne[0] = tensor->ne[0];
            rpc_tensor_desc.ne[1] = nrows_split;
            for (int d = 2; d < GGML_MAX_DIMS; d++) {
                rpc_tensor_desc.ne[d] = 1;
            }
            rpc_tensor_desc.data = extra->remote_ptrs[i];
            
            rpc_msg_get_tensor_req request;
            request.tensor = rpc_tensor_desc;
            request.offset = 0;
            request.size = split_size;
            
            bool status = send_rpc_cmd(extra->sockets[i], RPC_CMD_GET_TENSOR, &request, sizeof(request), 
                                       dst + split_offset, split_size);
            RPC_STATUS_ASSERT(status);
        }
    }
    
    GGML_UNUSED(buffer);
}

static void ggml_backend_rpc_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_split_buffer_context * ctx = (ggml_backend_rpc_split_buffer_context *)buffer->context;
    
    // Clear all remote buffers
    for (auto * extra : ctx->tensor_extras) {
        for (int i = 0; i < extra->n_endpoints; i++) {
            if (extra->remote_ptrs[i] == 0) continue;
            
            rpc_msg_buffer_clear_req request = {extra->remote_ptrs[i], value};
            bool status = send_rpc_cmd(extra->sockets[i], RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
            RPC_STATUS_ASSERT(status);
        }
    }
}

static ggml_backend_buffer_i ggml_backend_rpc_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL, // Split tensors can't be copied directly
    /* .clear           = */ ggml_backend_rpc_split_buffer_clear,
    /* .reset           = */ NULL,
};

// Buffer type interface for split buffers
static const char * ggml_backend_rpc_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_split_buffer_type_context * ctx = (ggml_backend_rpc_split_buffer_type_context *)buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_split_buffer_type_context * buft_ctx = (ggml_backend_rpc_split_buffer_type_context *)buft->context;
    
    ggml_backend_rpc_split_buffer_context * ctx = new ggml_backend_rpc_split_buffer_context{};
    ctx->n_endpoints = buft_ctx->n_endpoints;
    ctx->endpoints = buft_ctx->endpoints;
    ctx->devices = buft_ctx->devices;
    ctx->tensor_split = buft_ctx->tensor_split;
    
    // Establish connections to all endpoints
    for (int i = 0; i < ctx->n_endpoints; i++) {
        auto sock = get_socket(ctx->endpoints[i].c_str());
        if (sock == nullptr) {
            GGML_LOG_ERROR("RPC split buffer: failed to connect to %s\n", ctx->endpoints[i].c_str());
            delete ctx;
            return nullptr;
        }
        ctx->sockets.push_back(sock);
    }
    
    return ggml_backend_buffer_init(buft, ggml_backend_rpc_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_rpc_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // Return maximum alignment required by any endpoint
    ggml_backend_rpc_split_buffer_type_context * ctx = (ggml_backend_rpc_split_buffer_type_context *)buft->context;
    
    size_t max_alignment = 1;
    for (int i = 0; i < ctx->n_endpoints; i++) {
        auto sock = get_socket(ctx->endpoints[i].c_str());
        if (sock) {
            size_t alignment = get_alignment(sock, ctx->devices[i]);
            max_alignment = std::max(max_alignment, alignment);
        }
    }
    return max_alignment;
}

static size_t ggml_backend_rpc_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ ggml_backend_rpc_split_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

// Check if a buffer type is an RPC split buffer
bool ggml_backend_buft_is_rpc_split(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_rpc_split_buffer_type_name;
}

// Create an RPC split buffer type
ggml_backend_buffer_type_t ggml_backend_rpc_split_buffer_type(
        const char ** endpoints, const uint32_t * devices, const float * tensor_split, int n_endpoints) {
    
    if (n_endpoints <= 0 || n_endpoints > GGML_RPC_SPLIT_MAX_ENDPOINTS) {
        GGML_LOG_ERROR("RPC split buffer: invalid number of endpoints: %d\n", n_endpoints);
        return nullptr;
    }
    
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    
    // Build a unique name for this split configuration
    std::string buft_name = "RPC_SPLIT[";
    for (int i = 0; i < n_endpoints; i++) {
        if (i > 0) buft_name += ",";
        buft_name += std::string(endpoints[i]) + ":" + std::to_string(devices[i]);
    }
    buft_name += "]";
    
    // Check if we already have this buffer type
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    
    // Note: We don't verify endpoint connectivity here since it can abort on failure.
    // Connectivity will be verified when the buffer is actually allocated.
    
    // Create the context
    ggml_backend_rpc_split_buffer_type_context * ctx = new ggml_backend_rpc_split_buffer_type_context{};
    ctx->n_endpoints = n_endpoints;
    ctx->name = buft_name;
    
    float split_sum = 0.0f;
    for (int i = 0; i < n_endpoints; i++) {
        ctx->endpoints.push_back(endpoints[i]);
        ctx->devices.push_back(devices[i]);
        ctx->tensor_split[i] = tensor_split ? tensor_split[i] : 1.0f;
        split_sum += ctx->tensor_split[i];
    }
    
    // Normalize tensor_split if needed
    if (split_sum > 0 && std::abs(split_sum - 1.0f) > 0.01f) {
        for (int i = 0; i < n_endpoints; i++) {
            ctx->tensor_split[i] /= split_sum;
        }
    }
    
    // Create the buffer type
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_split_buffer_type_interface,
        /* .device  = */ nullptr, // Split buffer doesn't belong to a single device
        /* .context = */ ctx
    };
    
    buft_map[buft_name] = buft;
    
    GGML_LOG_INFO("RPC split buffer type created: %s\n", buft_name.c_str());
    return buft;
}

// ============================================================================
// Expert-based splitting (for MoE expert tensors)
// ============================================================================

// Calculate expert split for a given endpoint (splits on dim 2 = n_expert)
// This keeps complete experts together on each endpoint
static void rpc_split_get_expert_range(int64_t * expert_low, int64_t * expert_high,
                                       int64_t n_expert, const float * tensor_split,
                                       int n_endpoints, int endpoint_id) {
    // Calculate cumulative split proportions (same logic as row split)
    float sum = 0.0f;
    for (int i = 0; i < n_endpoints; i++) {
        sum += tensor_split[i];
    }
    if (sum == 0.0f) {
        sum = (float)n_endpoints;
    }
    
    float cumulative = 0.0f;
    for (int i = 0; i < endpoint_id; i++) {
        cumulative += tensor_split[i] / sum;
    }
    
    *expert_low = (int64_t)(n_expert * cumulative);
    
    if (endpoint_id == n_endpoints - 1) {
        *expert_high = n_expert;
    } else {
        cumulative += tensor_split[endpoint_id] / sum;
        *expert_high = (int64_t)(n_expert * cumulative);
    }
    
    // Ensure at least one expert per endpoint if possible
    if (*expert_high == *expert_low && endpoint_id < n_endpoints - 1 && *expert_low < n_expert) {
        *expert_high = *expert_low + 1;
    }
}

// Get the endpoint that owns a specific expert ID
static int rpc_split_get_expert_endpoint(int64_t expert_id, int64_t n_expert, 
                                         const float * tensor_split, int n_endpoints) {
    for (int i = 0; i < n_endpoints; i++) {
        int64_t expert_low, expert_high;
        rpc_split_get_expert_range(&expert_low, &expert_high, n_expert, tensor_split, n_endpoints, i);
        if (expert_id >= expert_low && expert_id < expert_high) {
            return i;
        }
    }
    return n_endpoints - 1;  // Fallback to last endpoint
}

// Check if tensor is a 3D MoE expert tensor (shape: [embd, ff, n_expert])
static bool rpc_split_is_expert_tensor(const ggml_tensor * tensor) {
    // Expert tensors have 3+ dimensions with ne[2] > 1 (multiple experts)
    // and names matching ffn_*_exps pattern
    // Note: ggml_n_dims() returns the number of dimensions with ne[i] > 1
    if (ggml_n_dims(tensor) < 3 || tensor->ne[2] <= 1) {
        return false;
    }
    
    // Check name pattern: blk.N.ffn_*_exps
    const char * name = tensor->name;
    if (name == nullptr) {
        return false;
    }
    
    return (strstr(name, "ffn_gate_exps") != nullptr ||
            strstr(name, "ffn_up_exps") != nullptr ||
            strstr(name, "ffn_down_exps") != nullptr);
}

// ============================================================================
// Distributed MUL_MAT_ID orchestration
// ============================================================================

// Statistics for distributed MUL_MAT_ID execution
struct rpc_split_mul_mat_id_stats {
    std::array<uint64_t, GGML_RPC_SPLIT_MAX_ENDPOINTS> compute_us;  // Per-endpoint compute time
    std::array<int64_t, GGML_RPC_SPLIT_MAX_ENDPOINTS> experts_computed;  // Experts computed per endpoint
    uint64_t total_us;  // Total wall-clock time
    int endpoints_used;  // Number of endpoints that participated
};

// Compute distributed MUL_MAT_ID across multiple RPC endpoints
// Parameters:
//   as_tensor: Expert tensor (shape [ne0, ne1, n_expert]) with split buffer
//   b_tensor: Input tensor (not split, full data on each endpoint)
//   ids_tensor: Expert selection indices (shape [top_k, n_tokens])
//   output: Output tensor to store results
//   stats: Optional statistics output
// Returns: true on success, false on failure
static bool rpc_split_mul_mat_id(
        const ggml_tensor * as_tensor,
        const ggml_tensor * b_tensor,
        const ggml_tensor * ids_tensor,
        ggml_tensor * output,
        rpc_split_mul_mat_id_stats * stats = nullptr) {
    
    // Verify as_tensor is a split tensor
    ggml_backend_rpc_split_tensor_extra * extra = 
        (ggml_backend_rpc_split_tensor_extra *)as_tensor->extra;
    if (extra == nullptr || !extra->is_expert_split) {
        GGML_LOG_ERROR("[%s] as_tensor is not an expert-split tensor\n", __func__);
        return false;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    const int n_endpoints = extra->n_endpoints;
    const int64_t n_expert = as_tensor->ne[2];
    
    // Prepare b tensor data to send to all endpoints
    size_t b_size = ggml_nbytes(b_tensor);
    std::vector<uint8_t> b_data(b_size);
    // For split buffer orchestration, we need to gather b from host memory
    // In production, b_tensor would be a CPU tensor or we'd have a copy
    memcpy(b_data.data(), b_tensor->data, b_size);
    
    // Prepare ids tensor data
    size_t ids_size = ggml_nbytes(ids_tensor);
    std::vector<uint8_t> ids_data(ids_size);
    memcpy(ids_data.data(), ids_tensor->data, ids_size);
    
    // Prepare output accumulator (zero-initialized)
    size_t output_size = ggml_nbytes(output);
    std::vector<float> output_accum(output_size / sizeof(float), 0.0f);
    
    // Send requests to all endpoints in parallel (simplified: sequential here)
    // TODO: Use async I/O for true parallelism
    int endpoints_used = 0;
    
    for (int i = 0; i < n_endpoints; i++) {
        if (extra->remote_ptrs[i] == 0) {
            continue;  // No experts on this endpoint
        }
        
        int64_t expert_low = extra->expert_low[i];
        int64_t expert_high = extra->expert_high[i];
        
        // Build request
        rpc_msg_mul_mat_id_partial_req request;
        memset(&request, 0, sizeof(request));
        
        // as tensor descriptor (local portion on this endpoint)
        request.as.id = (uint64_t)as_tensor;
        request.as.type = as_tensor->type;
        request.as.buffer = extra->remote_ptrs[i];
        request.as.ne[0] = as_tensor->ne[0];
        request.as.ne[1] = as_tensor->ne[1];
        request.as.ne[2] = expert_high - expert_low;  // Local expert count
        request.as.ne[3] = 1;
        request.as.data = extra->remote_ptrs[i];
        snprintf(request.as.name, GGML_MAX_NAME, "%.50s_e%d", as_tensor->name, i);
        
        // b tensor descriptor (full)
        request.b.type = b_tensor->type;
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            request.b.ne[d] = b_tensor->ne[d];
        }
        
        // ids tensor descriptor (full)
        request.ids.type = ids_tensor->type;
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            request.ids.ne[d] = ids_tensor->ne[d];
        }
        
        // output tensor descriptor
        request.output.type = output->type;
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            request.output.ne[d] = output->ne[d];
        }
        
        request.expert_low = expert_low;
        request.expert_high = expert_high;
        request.device = 0;  // TODO: Get from endpoint context
        
        // Send request and data
        auto sock = extra->sockets[i];
        if (sock == nullptr) {
            GGML_LOG_ERROR("[%s] no socket for endpoint %d\n", __func__, i);
            continue;
        }
        
        bool status = send_rpc_cmd(sock, RPC_CMD_MUL_MAT_ID_PARTIAL, &request, sizeof(request));
        if (!status) {
            GGML_LOG_ERROR("[%s] failed to send request to endpoint %d\n", __func__, i);
            continue;
        }
        
        // Send b tensor data
        if (!send_data(sock->fd, b_data.data(), b_data.size())) {
            GGML_LOG_ERROR("[%s] failed to send b data to endpoint %d\n", __func__, i);
            continue;
        }
        
        // Send ids tensor data
        if (!send_data(sock->fd, ids_data.data(), ids_data.size())) {
            GGML_LOG_ERROR("[%s] failed to send ids data to endpoint %d\n", __func__, i);
            continue;
        }
        
        // Receive response
        rpc_msg_mul_mat_id_partial_rsp response;
        if (!recv_data(sock->fd, &response, sizeof(response))) {
            GGML_LOG_ERROR("[%s] failed to receive response from endpoint %d\n", __func__, i);
            continue;
        }
        
        if (!response.success) {
            GGML_LOG_ERROR("[%s] computation failed on endpoint %d\n", __func__, i);
            continue;
        }
        
        // Receive partial output and accumulate
        std::vector<float> partial_output(output_size / sizeof(float));
        if (!recv_data(sock->fd, partial_output.data(), output_size)) {
            GGML_LOG_ERROR("[%s] failed to receive output from endpoint %d\n", __func__, i);
            continue;
        }
        
        // Accumulate partial results
        for (size_t j = 0; j < partial_output.size(); j++) {
            output_accum[j] += partial_output[j];
        }
        
        // Record statistics
        if (stats) {
            stats->compute_us[i] = response.compute_us;
            stats->experts_computed[i] = expert_high - expert_low;
        }
        
        endpoints_used++;
        
        LOG_DBG("[%s] endpoint %d: experts %ld-%ld, compute_us=%lu\n", 
                __func__, i, expert_low, expert_high - 1, response.compute_us);
    }
    
    // Copy accumulated output to result tensor
    memcpy(output->data, output_accum.data(), output_size);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    if (stats) {
        stats->total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        stats->endpoints_used = endpoints_used;
    }
    
    LOG_DBG("[%s] distributed MUL_MAT_ID complete: %d endpoints, %lu us total\n", 
            __func__, endpoints_used, 
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    
    return endpoints_used > 0;
}

// Check if a MUL_MAT_ID operation can be computed distributedly
// Returns true if the 'as' tensor is split across multiple endpoints
static bool rpc_split_can_distribute_mul_mat_id(const ggml_tensor * as_tensor) {
    if (as_tensor == nullptr || as_tensor->extra == nullptr) {
        return false;
    }
    
    ggml_backend_rpc_split_tensor_extra * extra = 
        (ggml_backend_rpc_split_tensor_extra *)as_tensor->extra;
    
    return extra->is_expert_split && extra->n_endpoints > 1;
}
