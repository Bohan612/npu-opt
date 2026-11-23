#include <stddef.h>
#include <stdint.h>

static void bm_memset(void *dst, uint8_t val, size_t n) {
  uint8_t *p = (uint8_t *)dst;
  for (size_t i = 0; i < n; ++i) {
    p[i] = val;
  }
}

static void bm_memcpy(void *dst, const void *src, size_t n) {
  uint8_t *d = (uint8_t *)dst;
  const uint8_t *s = (const uint8_t *)src;
  for (size_t i = 0; i < n; ++i) {
    d[i] = s[i];
  }
}

// IEEE-754 binary16 <-> binary32 helpers (works on any C compiler without _Float16)
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t bits = 0;

  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Subnormal fp16 -> normalized fp32
      int e = -1;
      uint32_t m = mant;
      while ((m & 0x0400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x03ffu;
      bits = sign | ((uint32_t)(127 - 14 + e) << 23) | (m << 13);
    }
  } else if (exp == 31) {
    // Inf / NaN
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }

  float out = 0.0f;
  bm_memcpy(&out, &bits, sizeof(out));
  return out;
}

static uint16_t f32_to_f16(float val) {
  uint32_t bits = 0;
  bm_memcpy(&bits, &val, sizeof(bits));

  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127;
  uint32_t mant = bits & 0x007fffffu;

  if (exp == 128) {
    if (mant == 0) {
      return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | 0x7e00u | (mant >> 13));
  }

  if (exp > 15) {
    // Overflow to infinity
    return (uint16_t)(sign | 0x7c00u);
  }

  if (exp < -14) {
    if (exp < -24) {
      return (uint16_t)sign;
    }
    mant |= 0x00800000u;
    int shift = -14 - exp;
    uint32_t sub = mant >> (13 + shift);
    uint32_t rem = mant & ((1u << (13 + shift)) - 1u);
    uint32_t half_way = 1u << (12 + shift);
    if (rem > half_way || (rem == half_way && (sub & 1u))) {
      ++sub;
    }
    return (uint16_t)(sign | sub);
  }

  uint16_t h_exp = (uint16_t)((exp + 15) << 10);
  uint16_t h_mant = (uint16_t)(mant >> 13);
  uint32_t rem = mant & 0x1fffu;
  uint16_t res = (uint16_t)(sign | h_exp | h_mant);
  if (rem > 0x1000u || (rem == 0x1000u && (res & 1u))) {
    res = (uint16_t)(res + 1u);
  }
  return res;
}

static uint64_t g_dma_token = 1;

// Matches: declare i64 @npu_dma_push(ptr, ptr addrspace(1), i64)
uint64_t npu_dma_push(const void *src_ddr, void *dst_sram, int64_t num_bytes) {
  if (src_ddr && dst_sram && num_bytes > 0) {
    bm_memcpy(dst_sram, src_ddr, (size_t)num_bytes);
  }
  return g_dma_token++;
}

// Matches: declare void @npu_dma_wait(i64)
void npu_dma_wait(uint64_t token) {
  (void)token;
  __asm__ volatile("" ::: "memory");
}

void npu_cube_mma_impl(const uint16_t *a, const uint16_t *b, const uint16_t *c, uint16_t *out) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      float acc = f16_to_f32(c[i * 16 + j]);
      for (int k = 0; k < 16; ++k) {
        acc += f16_to_f32(a[i * 16 + k]) * f16_to_f32(b[k * 16 + j]);
      }
      out[i * 16 + j] = f32_to_f16(acc);
    }
  }
}

extern void matmul_128x128x128_e2e(
    void *a_alloc, void *a_align, int64_t a_off, int64_t a_s0, int64_t a_s1, int64_t a_st0, int64_t a_st1,
    void *b_alloc, void *b_align, int64_t b_off, int64_t b_s0, int64_t b_s1, int64_t b_st0, int64_t b_st1,
    void *c_alloc, void *c_align, int64_t c_off, int64_t c_s0, int64_t c_s1, int64_t c_st0, int64_t c_st1);

static _Alignas(64) uint16_t A[128 * 128];
static _Alignas(64) uint16_t B[128 * 128];
_Alignas(64) uint16_t C[128 * 128];

int baremetal_run(void) {
  bm_memset(C, 0, sizeof(C));
  uint16_t one_f16 = f32_to_f16(1.0f);
  for (int i = 0; i < 128 * 128; ++i) {
    A[i] = one_f16;
    B[i] = one_f16;
  }

  matmul_128x128x128_e2e(
      A, A, 0, 128, 128, 128, 1,
      B, B, 0, 128, 128, 128, 1,
      C, C, 0, 128, 128, 128, 1);

  for (int i = 0; i < 128 * 128; ++i) {
    float val = f16_to_f32(C[i]);
    if (val != 128.0f) {
      return 1;
    }
  }
  return 0;
}
