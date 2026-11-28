#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKENS 4096
#define DMA_BYTES_PER_CYCLE 16
#define CUBE_MMA_CYCLES     64
#define VEC_ADD_CYCLES      16
#define VEC_RELU_CYCLES     8

typedef struct {
  uint64_t complete_cycle;
  uint64_t bytes;
  int is_store;
} DmaTokenEntry;

static DmaTokenEntry g_tokens[MAX_TOKENS];
static uint64_t g_next_token = 1;

static uint64_t g_dma_rd_clk = 0;
static uint64_t g_dma_wr_clk = 0;
static uint64_t g_cube_clk   = 0;
static uint64_t g_vec_clk    = 0;

static uint64_t g_total_dma_rd_cycles = 0;
static uint64_t g_total_dma_wr_cycles = 0;
static uint64_t g_total_cube_cycles   = 0;
static uint64_t g_total_vec_cycles    = 0;

// IEEE-754 binary16 <-> binary32 helpers
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) { bits = sign; } else {
      int e = -1; uint32_t m = mant;
      while ((m & 0x0400u) == 0) { m <<= 1; --e; }
      m &= 0x03ffu;
      bits = sign | ((uint32_t)(127 - 14 + e) << 23) | (m << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float out = 0.0f;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

static uint16_t f32_to_f16(float val) {
  uint32_t bits = 0;
  memcpy(&bits, &val, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127;
  uint32_t mant = bits & 0x007fffffu;
  if (exp == 128) {
    if (mant == 0) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | 0x7e00u | (mant >> 13));
  }
  if (exp > 15) return (uint16_t)(sign | 0x7c00u);
  if (exp < -14) {
    if (exp < -24) return (uint16_t)sign;
    mant |= 0x00800000u;
    int shift = -14 - exp;
    uint32_t sub = mant >> (13 + shift);
    uint32_t rem = mant & ((1u << (13 + shift)) - 1u);
    uint32_t half_way = 1u << (12 + shift);
    if (rem > half_way || (rem == half_way && (sub & 1u))) ++sub;
    return (uint16_t)(sign | sub);
  }
  uint16_t h_exp = (uint16_t)((exp + 15) << 10);
  uint16_t h_mant = (uint16_t)(mant >> 13);
  uint32_t rem = mant & 0x1fffu;
  uint16_t res = (uint16_t)(sign | h_exp | h_mant);
  if (rem > 0x1000u || (rem == 0x1000u && (res & 1u))) res = (uint16_t)(res + 1u);
  return res;
}

uint64_t npu_dma_load_2d(const uint16_t *dram_src, uint16_t *sram_dst,
                         int64_t rows, int64_t cols, int64_t dram_stride) {
  for (int64_t r = 0; r < rows; ++r) {
    memcpy(sram_dst + r * cols, dram_src + r * dram_stride, (size_t)cols * sizeof(uint16_t));
  }
  uint64_t bytes = (uint64_t)(rows * cols * sizeof(uint16_t));
  uint64_t cycles = (bytes + DMA_BYTES_PER_CYCLE - 1) / DMA_BYTES_PER_CYCLE;
  g_dma_rd_clk += cycles;
  g_total_dma_rd_cycles += cycles;
  uint64_t tok = g_next_token++;
  g_tokens[tok % MAX_TOKENS] = (DmaTokenEntry){g_dma_rd_clk, bytes, 0};
  return tok;
}

uint64_t npu_dma_store_2d(const uint16_t *sram_src, uint16_t *dram_dst,
                          int64_t rows, int64_t cols, int64_t dram_stride) {
  for (int64_t r = 0; r < rows; ++r) {
    memcpy(dram_dst + r * dram_stride, sram_src + r * cols, (size_t)cols * sizeof(uint16_t));
  }
  uint64_t bytes = (uint64_t)(rows * cols * sizeof(uint16_t));
  uint64_t cycles = (bytes + DMA_BYTES_PER_CYCLE - 1) / DMA_BYTES_PER_CYCLE;
  if (g_dma_wr_clk < g_vec_clk) g_dma_wr_clk = g_vec_clk;
  g_dma_wr_clk += cycles;
  g_total_dma_wr_cycles += cycles;
  uint64_t tok = g_next_token++;
  g_tokens[tok % MAX_TOKENS] = (DmaTokenEntry){g_dma_wr_clk, bytes, 1};
  return tok;
}

void npu_dma_wait(uint64_t token) {
  DmaTokenEntry e = g_tokens[token % MAX_TOKENS];
  if (!e.is_store) {
    if (g_cube_clk < e.complete_cycle) g_cube_clk = e.complete_cycle;
    if (g_vec_clk  < e.complete_cycle) g_vec_clk  = e.complete_cycle;
  }
}

void npu_cube_mma(const uint16_t *a, const uint16_t *b,
                  const uint16_t *c, uint16_t *out) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      float acc = f16_to_f32(c[i * 16 + j]);
      for (int k = 0; k < 16; ++k) {
        acc += f16_to_f32(a[i * 16 + k]) * f16_to_f32(b[k * 16 + j]);
      }
      out[i * 16 + j] = f32_to_f16(acc);
    }
  }
  g_cube_clk += CUBE_MMA_CYCLES;
  g_total_cube_cycles += CUBE_MMA_CYCLES;
}

void npu_cube_mma_c_harness(const uint16_t *a, const uint16_t *b, uint16_t *out) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      float acc = f16_to_f32(out[i * 16 + j]);
      for (int k = 0; k < 16; ++k) {
        acc += f16_to_f32(a[i * 16 + k]) * f16_to_f32(b[k * 16 + j]);
      }
      out[i * 16 + j] = f32_to_f16(acc);
    }
  }
  g_cube_clk += CUBE_MMA_CYCLES;
  g_total_cube_cycles += CUBE_MMA_CYCLES;
}

void npu_vec_add_c_harness(uint16_t *target, const uint16_t *bias) {
  if (g_vec_clk < g_cube_clk) g_vec_clk = g_cube_clk;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      float acc = f16_to_f32(target[i * 16 + j]);
      acc += f16_to_f32(bias[j]);
      target[i * 16 + j] = f32_to_f16(acc);
    }
  }
  g_vec_clk += VEC_ADD_CYCLES;
  g_total_vec_cycles += VEC_ADD_CYCLES;
}

void npu_vec_relu_c_harness(uint16_t *target) {
  if (g_vec_clk < g_cube_clk) g_vec_clk = g_cube_clk;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      float val = f16_to_f32(target[i * 16 + j]);
      if (val < 0.0f) val = 0.0f;
      target[i * 16 + j] = f32_to_f16(val);
    }
  }
  g_vec_clk += VEC_RELU_CYCLES;
  g_total_vec_cycles += VEC_RELU_CYCLES;
}

extern void matmul_bias(
    void* A_alloc, void* A_aligned, int64_t A_offset, int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    void* B_alloc, void* B_aligned, int64_t B_offset, int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    void* C_alloc, void* C_aligned, int64_t C_offset, int64_t C_size0, int64_t C_size1, int64_t C_stride0, int64_t C_stride1,
    void* bias_alloc, void* bias_aligned, int64_t bias_offset, int64_t bias_size0, int64_t bias_stride0
);

extern void matmul_bias_relu(
    void* A_alloc, void* A_aligned, int64_t A_offset, int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    void* B_alloc, void* B_aligned, int64_t B_offset, int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    void* C_alloc, void* C_aligned, int64_t C_offset, int64_t C_size0, int64_t C_size1, int64_t C_stride0, int64_t C_stride1,
    void* bias_alloc, void* bias_aligned, int64_t bias_offset, int64_t bias_size0, int64_t bias_stride0
);

int main() {
    printf("Initializing NPU Memory buffers (128x128)...\n");

    uint16_t *A = malloc(128 * 128 * sizeof(uint16_t));
    uint16_t *B = malloc(128 * 128 * sizeof(uint16_t));
    uint16_t *C = malloc(128 * 128 * sizeof(uint16_t));
    uint16_t *bias = malloc(128 * sizeof(uint16_t));

    uint16_t one_f16 = f32_to_f16(1.0f);
    for (int i = 0; i < 128 * 128; ++i) {
        A[i] = one_f16;
        B[i] = one_f16;
        C[i] = 0;
    }
    for (int i = 0; i < 128; ++i) bias[i] = one_f16;

    printf("Dispatching Execution to NPU MLIR binary...\n");
    
    for (int i = 0; i < 128; ++i) bias[i] = (i % 2 == 0) ? one_f16 : f32_to_f16(-200.0f);

    printf("Dispatching Execution to NPU MLIR binary (matmul_bias_relu)...\n");

    matmul_bias_relu(
        A, A, 0, 128, 128, 128, 1,
        B, B, 0, 128, 128, 128, 1,
        C, C, 0, 128, 128, 128, 1,
        bias, bias, 0, 128, 1
    );

    printf("C[0] (positive bias +1.0 -> 129.0): %.1f (raw=0x%04x)\n", f16_to_f32(C[0]), C[0]);
    printf("C[1] (negative bias -200.0 -> clamped to 0.0): %.1f (raw=0x%04x)\n", f16_to_f32(C[1]), C[1]);

    printf("NPU Execution Complete!\n\n");
    
    uint64_t total_compute = g_total_cube_cycles + g_total_vec_cycles;
    uint64_t total_dma     = g_total_dma_rd_cycles;
    uint64_t makespan      = g_dma_wr_clk > g_cube_clk ? g_dma_wr_clk : g_cube_clk;

    printf("=== NPU Pipeline & Ratio Report ===\n");
    printf("Cube MMA Cycles  : %llu\n", (unsigned long long)g_total_cube_cycles);
    printf("Vec Add Cycles   : %llu\n", (unsigned long long)g_total_vec_cycles);
    printf("DMA Load Cycles  : %llu\n", (unsigned long long)g_total_dma_rd_cycles);
    printf("DMA Store Cycles : %llu\n", (unsigned long long)g_total_dma_wr_cycles);
    printf("T_compute : T_dma = %.2f : 1.00\n", (double)total_compute / (double)total_dma);
    printf("End-to-End Makespan: %llu cycles (Ideal Unoverlapped: %llu cycles)\n",
           (unsigned long long)makespan,
           (unsigned long long)(total_compute + g_total_dma_rd_cycles + g_total_dma_wr_cycles));

    printf("C[0] verification value: %.1f (raw=0x%04x)\n", f16_to_f32(C[0]), C[0]);

    free(A); free(B); free(C); free(bias);
    return 0;
}
