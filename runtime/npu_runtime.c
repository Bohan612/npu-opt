#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Exact 512-byte ABI match for LLVM's [16 x <16 x half>]
typedef struct {
  uint16_t data[16][16];
} CubeTile16x16;

static uint16_t f32_to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000u;
  int exp = ((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = (x >> 13) & 0x03ffu;
  if (exp <= 0) return (uint16_t)sign;
  if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
  return (uint16_t)(sign | (exp << 10) | mant);
}

static float f16_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t x = (exp == 0) ? sign : (sign | ((exp + 112) << 23) | (mant << 13));
  float f;
  memcpy(&f, &x, sizeof(f));
  return f;
}

static int64_t next_dma_token = 1;
static int dma_push_count = 0, dma_wait_count = 0, mma_call_count = 0;

int64_t npu_dma_push(void *src_ddr, void *dst_sram, int64_t num_bytes) {
  int64_t tok = next_dma_token++;
  dma_push_count++;
  printf("[NPU-HW DMA]  Push #%d: DDR(%p) -> SRAM(%p), %ld bytes | Token = %ld\n",
         dma_push_count, src_ddr, dst_sram, (long)num_bytes, (long)tok);
  memcpy(dst_sram, src_ddr, (size_t)num_bytes);
  return tok;
}

void npu_dma_wait(int64_t token) {
  dma_wait_count++;
  printf("[NPU-HW SYNC] Wait #%d: Token %ld retired\n", dma_wait_count, (long)token);
}

CubeTile16x16 npu_cube_mma(CubeTile16x16 lhs, CubeTile16x16 rhs, CubeTile16x16 acc) {
  mma_call_count++;
  printf("[NPU-HW CUBE] Executing 16x16x16 f16 MMA #%d\n", mma_call_count);
  CubeTile16x16 out;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      float sum = f16_to_f32(acc.data[i][j]);
      for (int k = 0; k < 16; ++k)
        sum += f16_to_f32(lhs.data[i][k]) * f16_to_f32(rhs.data[k][j]);
      out.data[i][j] = f32_to_f16(sum);
    }
  }
  return out;
}

extern CubeTile16x16 npu_to_llvm_pipeline(
    uint16_t *, uint16_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
    uint16_t *, uint16_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
    CubeTile16x16, CubeTile16x16, CubeTile16x16);

int main(void) {
  const int N = 64 * 64;
  uint16_t *ddr = aligned_alloc(64, N * sizeof(uint16_t));
  uint16_t *sram = aligned_alloc(64, N * sizeof(uint16_t));
  for (int i = 0; i < N; ++i) {
    ddr[i] = f32_to_f16((float)(i % 7));
    sram[i] = 0;
  }

  CubeTile16x16 A, B, C;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      A.data[i][j] = f32_to_f16(2.0f);
      B.data[i][j] = f32_to_f16((i == j) ? 1.0f : 0.0f);
      C.data[i][j] = f32_to_f16(1.5f);
    }
  }

  printf("=== Launching MLIR-Compiled NPU Kernel ===\n");
  CubeTile16x16 D = npu_to_llvm_pipeline(
      ddr, ddr, 0, 64, 64, 64, 1,
      sram, sram, 0, 64, 64, 64, 1,
      A, B, C);

  int errors = memcmp(ddr, sram, N * sizeof(uint16_t)) != 0;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
      if (fabsf(f16_to_f32(D.data[i][j]) - 3.5f) > 1e-2f) errors++;

  printf("=== Verification Complete ===\n");
  printf("DMA Pushes: %d | DMA Waits: %d | Cube MMAs: %d | D[0][0] = %.2f | Errors = %d\n",
         dma_push_count, dma_wait_count, mma_call_count, f16_to_f32(D.data[0][0]), errors);
  free(ddr);
  free(sram);
  return errors != 0;
}
