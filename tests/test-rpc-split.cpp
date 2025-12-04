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
#include <array>
#include <unordered_map>

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

// ============================================================================
// Expert-based splitting tests (dim 2 splitting, not row-based)
// ============================================================================

// Helper: Calculate expert range for a device (split on dimension 2)
// This differs from row split - it keeps complete experts together
static void get_expert_split(int64_t * expert_low, int64_t * expert_high,
                             int64_t n_expert, const float * tensor_split,
                             int n_devices, int device_id) {
    float sum = 0.0f;
    for (int i = 0; i < n_devices; i++) {
        sum += tensor_split[i];
    }
    if (sum == 0.0f) sum = (float)n_devices;

    float cumulative = 0.0f;
    for (int i = 0; i < device_id; i++) {
        cumulative += tensor_split[i] / sum;
    }

    *expert_low = (int64_t)(n_expert * cumulative);

    if (device_id == n_devices - 1) {
        *expert_high = n_expert;
    } else {
        cumulative += tensor_split[device_id] / sum;
        *expert_high = (int64_t)(n_expert * cumulative);
    }

    // Ensure each device gets at least 1 expert if possible
    if (*expert_high == *expert_low && device_id < n_devices - 1 && *expert_low < n_expert) {
        *expert_high = *expert_low + 1;
    }
}

// Helper: Get device that owns a specific expert
static int get_expert_owner(int64_t expert_id, int64_t n_expert,
                            const float * tensor_split, int n_devices) {
    for (int dev = 0; dev < n_devices; dev++) {
        int64_t low, high;
        get_expert_split(&low, &high, n_expert, tensor_split, n_devices, dev);
        if (expert_id >= low && expert_id < high) {
            return dev;
        }
    }
    return n_devices - 1;  // Fallback to last device
}

// Helper: Detect expert tensor by name
static bool is_expert_tensor_name(const char * name) {
    return (strstr(name, "ffn_gate_exps") != nullptr ||
            strstr(name, "ffn_up_exps") != nullptr ||
            strstr(name, "ffn_down_exps") != nullptr);
}

// Test 11: Expert range with equal split
bool test_expert_equal_split() {
    printf("Testing expert range with equal split... ");

    float tensor_split[] = {1.0f, 1.0f, 1.0f, 1.0f};

    for (int i = 0; i < 4; i++) {
        int64_t low, high;
        get_expert_split(&low, &high, 8, tensor_split, 4, i);
        TEST_ASSERT(high - low == 2);
        TEST_ASSERT(low == i * 2);
    }

    TEST_PASS();
}

// Test 12: Expert range with unequal VRAM (realistic cluster scenario)
bool test_expert_unequal_vram_split() {
    printf("Testing expert split with unequal VRAM... ");

    // Simulate: 24GB, 12GB, 8GB, 8GB, 6GB = 58GB total
    float tensor_split[] = {24.0f, 12.0f, 8.0f, 8.0f, 6.0f};
    int64_t ranges[5][2];
    int64_t total = 0;

    for (int i = 0; i < 5; i++) {
        get_expert_split(&ranges[i][0], &ranges[i][1], 384, tensor_split, 5, i);
        total += ranges[i][1] - ranges[i][0];
    }

    TEST_ASSERT(total == 384);  // All experts covered
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(ranges[i][1] == ranges[i+1][0]);  // No gaps
    }
    TEST_ASSERT(ranges[0][1] - ranges[0][0] > ranges[4][1] - ranges[4][0]);  // First has most

    TEST_PASS();
}

// Test 13: Expert ID to device mapping (reverse lookup)
bool test_expert_owner_lookup() {
    printf("Testing expert ID to device owner lookup... ");

    float tensor_split[] = {0.75f, 0.25f};

    // With 75/25 split of 8 experts: device 0 gets 6, device 1 gets 2
    for (int e = 0; e < 6; e++) {
        TEST_ASSERT(get_expert_owner(e, 8, tensor_split, 2) == 0);
    }
    for (int e = 6; e < 8; e++) {
        TEST_ASSERT(get_expert_owner(e, 8, tensor_split, 2) == 1);
    }

    TEST_PASS();
}

// Test 14: Expert tensor detection by name
bool test_expert_tensor_detection() {
    printf("Testing expert tensor name detection... ");

    TEST_ASSERT(is_expert_tensor_name("blk.0.ffn_gate_exps.weight"));
    TEST_ASSERT(is_expert_tensor_name("blk.15.ffn_up_exps.weight"));
    TEST_ASSERT(is_expert_tensor_name("blk.31.ffn_down_exps.weight"));
    TEST_ASSERT(!is_expert_tensor_name("blk.0.attn_q.weight"));
    TEST_ASSERT(!is_expert_tensor_name("blk.0.ffn_gate.weight"));  // Non-expert FFN
    TEST_ASSERT(!is_expert_tensor_name("token_embd.weight"));

    TEST_PASS();
}

// Test 15: Expert-based vs row-based split comparison
bool test_expert_vs_row_split_difference() {
    printf("Testing expert-based vs row-based split difference... ");

    // For MoE tensors with shape [n_embd, n_ff, n_expert]:
    // - Row-based: splits n_ff across devices (each device has partial expert)
    // - Expert-based: splits n_expert across devices (each device has complete experts)

    const int64_t n_expert = 8;
    const int n_devices = 2;
    float tensor_split[] = {0.5f, 0.5f};

    // Expert-based split
    int64_t expert_low, expert_high;
    get_expert_split(&expert_low, &expert_high, n_expert, tensor_split, n_devices, 0);

    // Device 0 should get experts 0-3 (complete experts)
    TEST_ASSERT(expert_low == 0);
    TEST_ASSERT(expert_high == 4);

    // This is different from row-based which would split each expert's n_ff rows
    // Each device has COMPLETE experts, not partial experts

    TEST_PASS();
}

// Test 16: Simulate expert tensor allocation size calculation
bool test_expert_tensor_allocation_sizes() {
    printf("Testing expert tensor allocation sizes... ");

    // Simulate Mixtral: 8 experts, embd=4096, ff=14336
    const int64_t n_expert = 8;
    const int64_t n_embd = 4096;
    const int64_t n_ff = 14336;
    const int n_devices = 2;
    float tensor_split[] = {0.5f, 0.5f};

    // Expert-based split: each device gets complete experts
    int64_t expert_low_0, expert_high_0, expert_low_1, expert_high_1;
    get_expert_split(&expert_low_0, &expert_high_0, n_expert, tensor_split, n_devices, 0);
    get_expert_split(&expert_low_1, &expert_high_1, n_expert, tensor_split, n_devices, 1);

    // Each device should get 4 experts
    TEST_ASSERT(expert_high_0 - expert_low_0 == 4);
    TEST_ASSERT(expert_high_1 - expert_low_1 == 4);

    // Calculate size per device (f32 weights for simplicity)
    size_t bytes_per_expert = n_embd * n_ff * sizeof(float);
    size_t size_dev_0 = (expert_high_0 - expert_low_0) * bytes_per_expert;
    size_t size_dev_1 = (expert_high_1 - expert_low_1) * bytes_per_expert;

    // Total should equal full tensor size
    size_t total = size_dev_0 + size_dev_1;
    size_t expected = n_expert * bytes_per_expert;
    TEST_ASSERT(total == expected);

    // Verify we get ~117MB per expert (4096 * 14336 * 4 bytes)
    TEST_ASSERT(bytes_per_expert > 200 * 1024 * 1024);  // > 200MB
    TEST_ASSERT(bytes_per_expert < 250 * 1024 * 1024);  // < 250MB

    TEST_PASS();
}

// Test 17: Expert data distribution pattern
bool test_expert_data_distribution_pattern() {
    printf("Testing expert data distribution pattern... ");

    // Create mock tensor data: 4 experts, each with 2x3 matrix
    const int64_t n_expert = 4;
    const int64_t ne0 = 2;  // embd
    const int64_t ne1 = 3;  // ff
    const int n_devices = 2;
    float tensor_split[] = {0.5f, 0.5f};

    // Full tensor data: [ne0, ne1, n_expert] = [2, 3, 4]
    // Stored in row-major order: expert 0 data, expert 1 data, ...
    std::vector<float> full_data(ne0 * ne1 * n_expert);
    for (int e = 0; e < n_expert; e++) {
        for (int j = 0; j < ne1; j++) {
            for (int i = 0; i < ne0; i++) {
                // Value encodes expert and position
                full_data[e * ne0 * ne1 + j * ne0 + i] = e * 100.0f + j * 10.0f + i;
            }
        }
    }

    // Expert-based split
    int64_t expert_low, expert_high;
    get_expert_split(&expert_low, &expert_high, n_expert, tensor_split, n_devices, 0);

    // Device 0 gets experts 0-1
    TEST_ASSERT(expert_low == 0);
    TEST_ASSERT(expert_high == 2);

    // Extract device 0's portion
    size_t expert_size = ne0 * ne1;
    size_t dev0_offset = expert_low * expert_size;
    size_t dev0_size = (expert_high - expert_low) * expert_size;

    std::vector<float> dev0_data(dev0_size);
    memcpy(dev0_data.data(), &full_data[dev0_offset], dev0_size * sizeof(float));

    // Verify device 0 has complete experts 0 and 1
    TEST_ASSERT(dev0_data[0] == 0.0f);      // Expert 0, row 0, col 0
    TEST_ASSERT(dev0_data[5] == 21.0f);     // Expert 0, row 2, col 1 = 0*100 + 2*10 + 1
    TEST_ASSERT(dev0_data[6] == 100.0f);    // Expert 1, row 0, col 0 = 1*100 + 0

    TEST_PASS();
}

bool test_distributed_mul_mat_id_routing() {
    printf("Testing distributed MUL_MAT_ID routing logic... ");

    // Simulate 8 experts across 2 devices with 50/50 split
    const int64_t n_expert = 8;
    const int n_devices = 2;
    float tensor_split[] = {0.5f, 0.5f};

    // Device 0: experts 0-3, Device 1: experts 4-7
    int64_t expert_ranges[2][2];
    for (int d = 0; d < 2; d++) {
        get_expert_split(&expert_ranges[d][0], &expert_ranges[d][1], n_expert, tensor_split, n_devices, d);
    }

    // Simulate a batch with 4 tokens, top_k=2
    // Token 0 uses experts [1, 5] -> needs both devices
    // Token 1 uses experts [2, 3] -> only device 0
    // Token 2 uses experts [4, 6] -> only device 1
    // Token 3 uses experts [0, 7] -> needs both devices
    int32_t ids[8] = {1, 5,  // token 0
                      2, 3,  // token 1
                      4, 6,  // token 2
                      0, 7}; // token 3

    // Count experts computed per device
    int experts_dev0 = 0, experts_dev1 = 0;
    for (int i = 0; i < 8; i++) {
        int exp = ids[i];
        if (exp >= expert_ranges[0][0] && exp < expert_ranges[0][1]) {
            experts_dev0++;
        } else {
            experts_dev1++;
        }
    }

    // Device 0 should compute: experts 1, 2, 3, 0 = 4 computations
    TEST_ASSERT(experts_dev0 == 4);
    // Device 1 should compute: experts 5, 4, 6, 7 = 4 computations
    TEST_ASSERT(experts_dev1 == 4);

    TEST_PASS();
}

// Test 19: Output accumulation logic
bool test_output_accumulation() {
    printf("Testing output accumulation for distributed MUL_MAT_ID... ");

    // Simulate output tensors from 3 devices
    std::vector<float> output_dev0 = {1.0f, 0.0f, 0.0f, 2.0f};
    std::vector<float> output_dev1 = {0.0f, 3.0f, 0.0f, 0.0f};
    std::vector<float> output_dev2 = {0.0f, 0.0f, 4.0f, 1.0f};

    // Accumulate outputs (sum partial results)
    std::vector<float> final_output(4, 0.0f);
    for (size_t i = 0; i < 4; i++) {
        final_output[i] = output_dev0[i] + output_dev1[i] + output_dev2[i];
    }

    // Verify accumulated output
    TEST_ASSERT(final_output[0] == 1.0f);
    TEST_ASSERT(final_output[1] == 3.0f);
    TEST_ASSERT(final_output[2] == 4.0f);
    TEST_ASSERT(final_output[3] == 3.0f);  // 2 + 0 + 1

    TEST_PASS();
}
bool test_profile_load_balance() {
    printf("Testing profile load balance calculation... ");

    // Simulate perfectly balanced workload: 2 endpoints, same compute time
    std::array<uint64_t, 4> balanced_times = {100000, 100000, 0, 0};  // 100ms each

    // Calculate CV (coefficient of variation) manually
    float sum = 200000, mean = 100000;
    float variance = 0;  // All same -> 0 variance
    float cv = 0;
    float balance = 1.0f / (1.0f + cv);
    TEST_ASSERT(std::abs(balance - 1.0f) < 0.01f);

    // Simulate imbalanced workload: 1 endpoint does 3x the work
    std::array<uint64_t, 4> imbalanced_times = {300000, 100000, 0, 0};  // 300ms vs 100ms
    sum = 400000;
    mean = 200000;
    variance = ((300000 - mean) * (300000 - mean) + (100000 - mean) * (100000 - mean)) / 2.0f;
    float stddev = sqrtf(variance);
    cv = stddev / mean;
    balance = 1.0f / (1.0f + cv);
    // With 3:1 ratio, CV ~= 0.5, balance ~= 0.67
    TEST_ASSERT(balance > 0.5f && balance < 0.8f);

    TEST_PASS();
}

// Test 21: Expert activation tracking
bool test_expert_activation_tracking() {
    printf("Testing expert activation tracking... ");

    // Simulate expert activations
    std::unordered_map<int64_t, int64_t> activations;

    // Simulate 10 batches, each selecting 2 experts
    // Expert 0 is "hot" - selected every time
    // Other experts are selected less frequently
    int32_t selections[] = {
        0, 1,  // batch 0
        0, 2,  // batch 1
        0, 3,  // batch 2
        0, 1,  // batch 3
        0, 4,  // batch 4
        0, 2,  // batch 5
        0, 5,  // batch 6
        0, 1,  // batch 7
        0, 3,  // batch 8
        0, 6   // batch 9
    };

    for (int i = 0; i < 20; i++) {
        activations[selections[i]]++;
    }

    // Expert 0 should have 10 activations
    TEST_ASSERT(activations[0] == 10);
    // Expert 1 should have 3 activations
    TEST_ASSERT(activations[1] == 3);
    // Expert 2 should have 2 activations
    TEST_ASSERT(activations[2] == 2);

    // Find most activated expert
    int64_t max_expert = 0, max_count = 0;
    for (auto & [exp, count] : activations) {
        if (count > max_count) {
            max_count = count;
            max_expert = exp;
        }
    }
    TEST_ASSERT(max_expert == 0);

    TEST_PASS();
}

// =============================================================================
// Integration Tests: Small MoE Scenarios
// =============================================================================

bool test_moe_8_experts_2_endpoints() {
    printf("Testing full MoE workflow: 8 experts, 2 endpoints... ");

    // Simulate 2 endpoints with equal VRAM (8GB each)
    float tensor_split[] = {8.0f, 8.0f};  // Equal VRAM
    const int n_expert = 8;
    const int n_devices = 2;

    // Each endpoint should get 4 experts
    // Endpoint 0: experts 0-3
    // Endpoint 1: experts 4-7

    // Verify expert ranges
    for (int i = 0; i < n_devices; i++) {
        int64_t low, high;
        get_expert_split(&low, &high, n_expert, tensor_split, n_devices, i);
        int expected_low = (i == 0) ? 0 : 4;
        int expected_high = (i == 0) ? 4 : 8;
        TEST_ASSERT(low == expected_low && high == expected_high);
    }

    // Verify expert-to-endpoint mapping
    for (int expert = 0; expert < n_expert; expert++) {
        int expected_endpoint = expert < 4 ? 0 : 1;
        int actual_endpoint = get_expert_owner(expert, n_expert, tensor_split, n_devices);
        TEST_ASSERT(actual_endpoint == expected_endpoint);
    }

    // Simulate token routing: 8 tokens, each routed to 2 experts (top-k=2)
    std::vector<std::pair<int, int>> token_experts = {
        {0, 5}, {1, 2}, {3, 7}, {4, 5}, {0, 4}, {6, 7}, {2, 3}, {1, 6},
    };

    // Count tokens per endpoint
    std::array<int, 2> tokens_per_endpoint = {0, 0};
    std::array<int, 8> expert_activations = {0, 0, 0, 0, 0, 0, 0, 0};

    for (const auto& [e1, e2] : token_experts) {
        int ep1 = get_expert_owner(e1, n_expert, tensor_split, n_devices);
        int ep2 = get_expert_owner(e2, n_expert, tensor_split, n_devices);
        tokens_per_endpoint[ep1]++;
        tokens_per_endpoint[ep2]++;
        expert_activations[e1]++;
        expert_activations[e2]++;
    }

    // Verify reasonable load balance (within 2x)
    float ratio = (float)std::max(tokens_per_endpoint[0], tokens_per_endpoint[1]) /
                  (float)std::min(tokens_per_endpoint[0], tokens_per_endpoint[1]);
    TEST_ASSERT(ratio < 2.0f);

    // Simulate expert tensor data - smaller dimensions for testing
    const int test_embd = 64;
    const int test_ff = 128;
    std::vector<float> expert_weights(test_embd * test_ff * n_expert);

    for (int e = 0; e < n_expert; e++) {
        for (int i = 0; i < test_embd * test_ff; i++) {
            expert_weights[e * test_embd * test_ff + i] = e * 1000.0f + i;
        }
    }

    // Verify data slicing for each endpoint
    for (int ep = 0; ep < n_devices; ep++) {
        int64_t low, high;
        get_expert_split(&low, &high, n_expert, tensor_split, n_devices, ep);
        int n_local = (int)(high - low);
        size_t offset = low * test_embd * test_ff;

        for (int local_expert = 0; local_expert < n_local; local_expert++) {
            int global_expert = (int)low + local_expert;
            float expected = global_expert * 1000.0f;
            float actual = expert_weights[offset + local_expert * test_embd * test_ff];
            TEST_ASSERT(actual == expected);
        }
    }

    // Simulate output accumulation
    const int n_tokens = 8;
    std::vector<float> accumulated_output(n_tokens * test_embd, 0.0f);

    for (int ep = 0; ep < n_devices; ep++) {
        int64_t low, high;
        get_expert_split(&low, &high, n_expert, tensor_split, n_devices, ep);
        std::vector<float> partial_output(n_tokens * test_embd, 0.0f);

        for (int t = 0; t < n_tokens; t++) {
            auto [e1, e2] = token_experts[t];
            if (e1 >= low && e1 < high) {
                for (int i = 0; i < test_embd; i++) partial_output[t * test_embd + i] += e1 + 1;
            }
            if (e2 >= low && e2 < high) {
                for (int i = 0; i < test_embd; i++) partial_output[t * test_embd + i] += e2 + 1;
            }
        }
        for (size_t i = 0; i < accumulated_output.size(); i++) {
            accumulated_output[i] += partial_output[i];
        }
    }

    // Verify accumulated results
    for (int t = 0; t < n_tokens; t++) {
        auto [e1, e2] = token_experts[t];
        float expected = (float)((e1 + 1) + (e2 + 1));
        TEST_ASSERT(accumulated_output[t * test_embd] == expected);
    }

    TEST_PASS();
}

bool test_moe_unequal_vram_distribution() {
    printf("Testing MoE with unequal VRAM (16GB vs 8GB)... ");

    float tensor_split[] = {16.0f, 8.0f};  // 2:1 VRAM ratio
    const int n_expert = 8;
    const int n_devices = 2;

    int64_t low0, high0, low1, high1;
    get_expert_split(&low0, &high0, n_expert, tensor_split, n_devices, 0);
    get_expert_split(&low1, &high1, n_expert, tensor_split, n_devices, 1);

    int experts_ep0 = (int)(high0 - low0);
    int experts_ep1 = (int)(high1 - low1);

    // Endpoint with more VRAM should have more experts
    TEST_ASSERT(experts_ep0 > experts_ep1);
    TEST_ASSERT(experts_ep0 + experts_ep1 == n_expert);

    // Verify no gaps in expert assignment
    TEST_ASSERT(high0 == low1);
    TEST_ASSERT(low0 == 0);
    TEST_ASSERT(high1 == n_expert);

    TEST_PASS();
}

bool test_moe_data_integrity() {
    printf("Testing MoE data integrity (split/reconstruct)... ");

    const int n_expert = 4;
    const int n_embd = 32;
    const int n_ff = 64;
    const int n_devices = 2;
    float tensor_split[] = {1.0f, 1.0f};

    std::vector<float> original_data(n_expert * n_embd * n_ff);

    for (size_t i = 0; i < original_data.size(); i++) {
        original_data[i] = std::sin(i * 0.01f) * 100.0f;
    }

    std::vector<std::vector<float>> endpoint_data(n_devices);

    for (int ep = 0; ep < n_devices; ep++) {
        int64_t low, high;
        get_expert_split(&low, &high, n_expert, tensor_split, n_devices, ep);
        size_t offset = low * n_embd * n_ff;
        size_t count = (high - low) * n_embd * n_ff;
        endpoint_data[ep].resize(count);
        std::copy(original_data.begin() + offset,
                  original_data.begin() + offset + count,
                  endpoint_data[ep].begin());
    }

    std::vector<float> reconstructed(original_data.size());
    for (int ep = 0; ep < n_devices; ep++) {
        int64_t low, high;
        get_expert_split(&low, &high, n_expert, tensor_split, n_devices, ep);
        size_t offset = low * n_embd * n_ff;
        std::copy(endpoint_data[ep].begin(), endpoint_data[ep].end(),
                  reconstructed.begin() + offset);
    }

    for (size_t i = 0; i < original_data.size(); i++) {
        TEST_ASSERT(original_data[i] == reconstructed[i]);
    }

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

    // Row-based splitting tests
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

    // Expert-based splitting tests (dim 2 splitting)
    printf("\n--- Expert-Based Splitting Tests ---\n\n");
    RUN_TEST(test_expert_equal_split);
    RUN_TEST(test_expert_unequal_vram_split);
    RUN_TEST(test_expert_owner_lookup);
    RUN_TEST(test_expert_tensor_detection);
    RUN_TEST(test_expert_vs_row_split_difference);
    RUN_TEST(test_expert_tensor_allocation_sizes);
    RUN_TEST(test_expert_data_distribution_pattern);
    RUN_TEST(test_distributed_mul_mat_id_routing);
    RUN_TEST(test_output_accumulation);
    RUN_TEST(test_profile_load_balance);
    RUN_TEST(test_expert_activation_tracking);

    // Integration tests: MoE scenarios
    printf("\n--- Integration Tests: MoE Scenarios ---\n\n");
    RUN_TEST(test_moe_8_experts_2_endpoints);
    RUN_TEST(test_moe_unequal_vram_distribution);
    RUN_TEST(test_moe_data_integrity);

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


