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
    // Row ranges for each endpoint
    std::array<int64_t, GGML_RPC_SPLIT_MAX_ENDPOINTS> row_low;
    std::array<int64_t, GGML_RPC_SPLIT_MAX_ENDPOINTS> row_high;
    int n_endpoints;
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
    
    const int64_t nrows = ggml_nrows(tensor);
    
    ggml_backend_rpc_split_tensor_extra * extra = new ggml_backend_rpc_split_tensor_extra{};
    ctx->tensor_extras.push_back(extra);
    extra->n_endpoints = ctx->n_endpoints;
    
    // Allocate memory on each endpoint for its portion of the tensor
    for (int i = 0; i < ctx->n_endpoints; i++) {
        int64_t row_low, row_high;
        rpc_split_get_row_split(&row_low, &row_high, nrows, ctx->tensor_split.data(), ctx->n_endpoints, i);
        
        extra->row_low[i] = row_low;
        extra->row_high[i] = row_high;
        
        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            extra->remote_ptrs[i] = 0;
            continue;
        }
        
        // Calculate size for this endpoint's portion
        size_t size = nrows_split * ggml_row_size(tensor->type, tensor->ne[0]);
        
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
        LOG_DBG("[%s] endpoint %d: allocated %zu bytes at %p for rows %ld-%ld\n", 
                __func__, i, size, (void*)response.remote_ptr, row_low, row_high);
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
    
    const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
    const uint8_t * src = (const uint8_t *)data;
    
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
    
    GGML_UNUSED(buffer);
}

static void ggml_backend_rpc_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, 
                                                     void * data, size_t offset, size_t size) {
    // Gather data from all endpoints back into the full tensor
    GGML_ASSERT(offset == 0 && "split tensors must be read from offset 0");
    GGML_ASSERT(size == ggml_nbytes(tensor) && "split tensors must be read in their entirety");
    
    ggml_backend_rpc_split_tensor_extra * extra = (ggml_backend_rpc_split_tensor_extra *)tensor->extra;
    GGML_ASSERT(extra != nullptr);
    
    const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
    uint8_t * dst = (uint8_t *)data;
    
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
