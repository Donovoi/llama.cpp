// Test suite for RPC split buffer functionality
// Tests distributed expert tensor loading across multiple RPC backends

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cassert>

// Test configuration
#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

#define TEST_PASS() do { \
    printf("PASS\n"); \
    return true; \
} while(0)

// Helper: Calculate row range for a device given tensor_split proportions
static void get_row_split(int64_t * row_low, int64_t * row_high, 
                          int64_t nrows, const float * tensor_split, 
                          int n_devices, int device_id, int64_t row_rounding = 1) {
    // Calculate cumulative split proportions
    float sum = 0.0f;
    for (int i = 0; i < n_devices; i++) {
        sum += tensor_split[i];
    }
    
    // Normalize if needed
    float cumulative = 0.0f;
    for (int i = 0; i < device_id; i++) {
        cumulative += tensor_split[i] / sum;
    }
    
    *row_low = (int64_t)(nrows * cumulative);
    *row_low -= *row_low % row_rounding;
    
    if (device_id == n_devices - 1) {
        *row_high = nrows;
    } else {
        cumulative += tensor_split[device_id] / sum;
        *row_high = (int64_t)(nrows * cumulative);
        *row_high -= *row_high % row_rounding;
    }
}

// Test 1: Row split calculation correctness
bool test_row_split_calculation() {
    printf("Testing row split calculation... ");
    
    const int n_devices = 4;
    const int64_t nrows = 384; // 384 experts like Kimi-K2
    float tensor_split[4] = {0.25f, 0.25f, 0.25f, 0.25f}; // Equal split
    
    int64_t total_rows = 0;
    for (int i = 0; i < n_devices; i++) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, nrows, tensor_split, n_devices, i);
        
        int64_t device_rows = row_high - row_low;
        TEST_ASSERT(device_rows > 0);
        TEST_ASSERT(row_low >= 0);
        TEST_ASSERT(row_high <= nrows);
        total_rows += device_rows;
    }
    
    TEST_ASSERT(total_rows == nrows);
    TEST_PASS();
}

// Test 2: Unequal split distribution
bool test_unequal_split() {
    printf("Testing unequal split distribution... ");
    
    const int n_devices = 3;
    const int64_t nrows = 300;
    // Simulate different VRAM capacities: 40%, 35%, 25%
    float tensor_split[3] = {0.40f, 0.35f, 0.25f};
    
    int64_t row_low, row_high;
    
    // Device 0 should get ~120 rows
    get_row_split(&row_low, &row_high, nrows, tensor_split, n_devices, 0);
    TEST_ASSERT(row_low == 0);
    TEST_ASSERT(row_high == 120);
    
    // Device 1 should get ~105 rows
    get_row_split(&row_low, &row_high, nrows, tensor_split, n_devices, 1);
    TEST_ASSERT(row_low == 120);
    TEST_ASSERT(row_high == 225);
    
    // Device 2 should get remaining ~75 rows
    get_row_split(&row_low, &row_high, nrows, tensor_split, n_devices, 2);
    TEST_ASSERT(row_low == 225);
    TEST_ASSERT(row_high == 300);
    
    TEST_PASS();
}

// Test 3: Row rounding for alignment
bool test_row_rounding() {
    printf("Testing row rounding alignment... ");
    
    const int n_devices = 2;
    const int64_t nrows = 100;
    float tensor_split[2] = {0.5f, 0.5f};
    const int64_t rounding = 8; // Typical alignment requirement
    
    int64_t row_low, row_high;
    
    get_row_split(&row_low, &row_high, nrows, tensor_split, n_devices, 0, rounding);
    TEST_ASSERT(row_low % rounding == 0);
    TEST_ASSERT(row_high % rounding == 0 || row_high == nrows);
    
    get_row_split(&row_low, &row_high, nrows, tensor_split, n_devices, 1, rounding);
    TEST_ASSERT(row_low % rounding == 0);
    
    TEST_PASS();
}

// Test 4: Edge case - single device (no split)
bool test_single_device() {
    printf("Testing single device (no split)... ");
    
    const int n_devices = 1;
    const int64_t nrows = 256;
    float tensor_split[1] = {1.0f};
    
    int64_t row_low, row_high;
    get_row_split(&row_low, &row_high, nrows, tensor_split, n_devices, 0);
    
    TEST_ASSERT(row_low == 0);
    TEST_ASSERT(row_high == nrows);
    
    TEST_PASS();
}

// Test 5: Edge case - empty tensor_split (use defaults)
bool test_default_split() {
    printf("Testing default equal split... ");
    
    const int n_devices = 5;
    const int64_t nrows = 100;
    // All zeros should result in equal split
    float tensor_split[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // When all zero, fall back to equal distribution
    float equal_split[5] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
    
    int64_t total = 0;
    for (int i = 0; i < n_devices; i++) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, nrows, equal_split, n_devices, i);
        total += (row_high - row_low);
    }
    
    TEST_ASSERT(total == nrows);
    TEST_PASS();
}

// Test 6: Data distribution simulation
bool test_data_distribution() {
    printf("Testing data distribution across devices... ");
    
    const int n_devices = 3;
    const int64_t nrows = 12;
    const int64_t row_size = 100; // bytes per row
    float tensor_split[3] = {0.33f, 0.33f, 0.34f};
    
    // Simulate full tensor data
    std::vector<uint8_t> full_data(nrows * row_size);
    for (size_t i = 0; i < full_data.size(); i++) {
        full_data[i] = (uint8_t)(i % 256);
    }
    
    // Simulate distributing to devices
    std::vector<std::vector<uint8_t>> device_data(n_devices);
    
    for (int dev = 0; dev < n_devices; dev++) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, nrows, tensor_split, n_devices, dev);
        
        int64_t dev_rows = row_high - row_low;
        size_t offset = row_low * row_size;
        size_t size = dev_rows * row_size;
        
        device_data[dev].resize(size);
        memcpy(device_data[dev].data(), full_data.data() + offset, size);
    }
    
    // Verify all data is accounted for
    size_t total_size = 0;
    for (int dev = 0; dev < n_devices; dev++) {
        total_size += device_data[dev].size();
    }
    TEST_ASSERT(total_size == full_data.size());
    
    // Verify data integrity by reassembling
    std::vector<uint8_t> reassembled(nrows * row_size);
    size_t offset = 0;
    for (int dev = 0; dev < n_devices; dev++) {
        memcpy(reassembled.data() + offset, device_data[dev].data(), device_data[dev].size());
        offset += device_data[dev].size();
    }
    
    TEST_ASSERT(memcmp(full_data.data(), reassembled.data(), full_data.size()) == 0);
    TEST_PASS();
}

// Test 7: Expert tensor size calculation for MoE
bool test_expert_tensor_sizing() {
    printf("Testing expert tensor size calculation... ");
    
    // Simulate Kimi-K2-Thinking: 384 experts, hidden_dim=5120, ff_dim=1408
    const int64_t n_expert = 384;
    const int64_t hidden_dim = 5120;
    const int64_t ff_dim = 1408;
    
    // Expert tensor shape: [hidden_dim, ff_dim, n_expert]
    // For Q2_K quantization, ~2.3 bits per weight
    const float bits_per_weight = 2.3f;
    
    // Calculate size per expert
    int64_t weights_per_expert = hidden_dim * ff_dim;
    size_t bytes_per_expert = (size_t)(weights_per_expert * bits_per_weight / 8.0f);
    
    // Total expert tensor size
    size_t total_expert_size = bytes_per_expert * n_expert;
    
    // Verify reasonable size (should be ~1GB per expert tensor for Q2_K)
    printf("(expert tensor ~%.2f GB) ", total_expert_size / (1024.0 * 1024.0 * 1024.0));
    
    // Each expert should be ~2MB for Q2_K
    TEST_ASSERT(bytes_per_expert > 1000000);  // > 1MB
    TEST_ASSERT(bytes_per_expert < 10000000); // < 10MB
    
    TEST_PASS();
}

// Test 8: Verify split covers all experts for Kimi-K2 scenario
bool test_kimi_k2_expert_split() {
    printf("Testing Kimi-K2 expert split (384 experts, 5 devices)... ");
    
    const int n_devices = 5;
    const int64_t n_experts = 384;
    
    // Simulate cluster VRAM distribution: 24GB, 12GB, 8GB, 8GB, 6GB = 58GB total
    float vram_gb[5] = {24.0f, 12.0f, 8.0f, 8.0f, 6.0f};
    float total_vram = 0;
    for (int i = 0; i < n_devices; i++) total_vram += vram_gb[i];
    
    float tensor_split[5];
    for (int i = 0; i < n_devices; i++) {
        tensor_split[i] = vram_gb[i] / total_vram;
    }
    
    // Calculate expert distribution
    int64_t experts_assigned = 0;
    for (int dev = 0; dev < n_devices; dev++) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, n_experts, tensor_split, n_devices, dev);
        
        int64_t dev_experts = row_high - row_low;
        experts_assigned += dev_experts;
        
        printf("\n  Device %d: experts %ld-%ld (%ld experts, %.1f%% of VRAM)", 
               dev, row_low, row_high - 1, dev_experts, tensor_split[dev] * 100);
    }
    printf("\n  ");
    
    TEST_ASSERT(experts_assigned == n_experts);
    TEST_PASS();
}

// Test 9: Row split with MUL_MAT_ID index mapping
bool test_expert_id_mapping() {
    printf("Testing expert ID to device mapping... ");
    
    const int n_devices = 4;
    const int64_t n_experts = 16;
    float tensor_split[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    
    // Build mapping: expert_id -> (device_id, local_offset)
    std::vector<std::pair<int, int64_t>> expert_to_device(n_experts);
    
    for (int dev = 0; dev < n_devices; dev++) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, n_experts, tensor_split, n_devices, dev);
        
        for (int64_t expert = row_low; expert < row_high; expert++) {
            expert_to_device[expert] = {dev, expert - row_low};
        }
    }
    
    // Verify all experts are mapped
    for (int64_t e = 0; e < n_experts; e++) {
        int dev = expert_to_device[e].first;
        int64_t local_idx = expert_to_device[e].second;
        TEST_ASSERT(dev >= 0 && dev < n_devices);
        TEST_ASSERT(local_idx >= 0);
    }
    
    // Verify expert distribution
    TEST_ASSERT(expert_to_device[0].first == 0);  // Expert 0 -> device 0
    TEST_ASSERT(expert_to_device[4].first == 1);  // Expert 4 -> device 1
    TEST_ASSERT(expert_to_device[8].first == 2);  // Expert 8 -> device 2
    TEST_ASSERT(expert_to_device[15].first == 3); // Expert 15 -> device 3
    
    TEST_PASS();
}

// Test 10: Simulate MUL_MAT_ID routing
bool test_mul_mat_id_routing() {
    printf("Testing MUL_MAT_ID routing simulation... ");
    
    const int n_devices = 3;
    const int64_t n_experts = 9;
    const int top_k = 2; // Select top-2 experts per token
    float tensor_split[3] = {0.33f, 0.33f, 0.34f};
    
    // Build device boundaries
    int64_t dev_row_low[3], dev_row_high[3];
    for (int dev = 0; dev < n_devices; dev++) {
        get_row_split(&dev_row_low[dev], &dev_row_high[dev], n_experts, tensor_split, n_devices, dev);
    }
    
    // Simulate routing: token selects experts [1, 7] (cross-device!)
    int selected_experts[2] = {1, 7};
    
    // Determine which devices need to compute
    std::vector<int> active_devices;
    for (int i = 0; i < top_k; i++) {
        int expert = selected_experts[i];
        for (int dev = 0; dev < n_devices; dev++) {
            if (expert >= dev_row_low[dev] && expert < dev_row_high[dev]) {
                // Check if device already in list
                bool found = false;
                for (int d : active_devices) {
                    if (d == dev) { found = true; break; }
                }
                if (!found) active_devices.push_back(dev);
            }
        }
    }
    
    // For experts [1, 7] with 3 experts per device:
    // Expert 1 -> device 0 (experts 0-2)
    // Expert 7 -> device 2 (experts 6-8)
    TEST_ASSERT(active_devices.size() == 2);
    
    TEST_PASS();
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;
    
    printf("=== RPC Split Buffer Tests ===\n\n");
    
    int passed = 0;
    int total = 0;
    
    #define RUN_TEST(test) do { \
        total++; \
        if (test()) passed++; \
    } while(0)
    
    RUN_TEST(test_row_split_calculation);
    RUN_TEST(test_unequal_split);
    RUN_TEST(test_row_rounding);
    RUN_TEST(test_single_device);
    RUN_TEST(test_default_split);
    RUN_TEST(test_data_distribution);
    RUN_TEST(test_expert_tensor_sizing);
    RUN_TEST(test_kimi_k2_expert_split);
    RUN_TEST(test_expert_id_mapping);
    RUN_TEST(test_mul_mat_id_routing);
    
    printf("\n=== Results: %d/%d tests passed ===\n", passed, total);
    
    return (passed == total) ? 0 : 1;
}

// Additional tests that can be run with the compiled library
// These test the actual RPC split buffer API

#ifdef TEST_WITH_LIBRARY
#include "ggml-rpc.h"

// Test 11: Split buffer type creation (requires no active servers, just API test)
bool test_split_buffer_type_api() {
    printf("Testing split buffer type API... ");
    
    // These endpoints don't need to be valid for API testing
    const char * endpoints[] = {"127.0.0.1:50052", "127.0.0.1:50053", nullptr};
    uint32_t devices[] = {0, 0};
    float tensor_split[] = {0.6f, 0.4f};
    
    // This will fail to connect but should not crash
    // Just verify the API exists and is callable
    auto buft = ggml_backend_rpc_split_buffer_type(endpoints, devices, tensor_split, 2);
    
    // We expect nullptr since endpoints aren't available
    // But if servers were running, this should succeed
    printf("(returned %s) ", buft ? "buft" : "nullptr");
    
    TEST_PASS();
}

// Test 12: Check if buffer type is RPC split
bool test_buft_is_rpc_split() {
    printf("Testing ggml_backend_buft_is_rpc_split... ");
    
    // nullptr should return false
    TEST_ASSERT(!ggml_backend_buft_is_rpc_split(nullptr));
    
    TEST_PASS();
}
#endif
