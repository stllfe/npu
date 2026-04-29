#ifndef OPS_REG_MAIN_H
#define OPS_REG_MAIN_H

#include <stdint.h>

typedef struct {
  uint32_t mt[624];
  int index;
} Mt19937;

static inline void mt_seed(Mt19937 *rng, uint32_t seed) {
  rng->mt[0] = seed;
  for (int i = 1; i < 624; i++) {
    rng->mt[i] = 1812433253U * (rng->mt[i-1] ^ (rng->mt[i-1] >> 30)) + (uint32_t)i;
  }
  rng->index = 624;
}

static inline uint32_t mt_extract(Mt19937 *rng) {
  const uint32_t mag01[2] = {0U, 0x9908b0dfU};
  if (rng->index >= 624) {
    int kk;
    for (kk = 0; kk < 624 - 397; kk++) {
      uint32_t y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk+1] & 0x7fffffffU);
      rng->mt[kk] = rng->mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1U];
    }
    for (; kk < 623; kk++) {
      uint32_t y = (rng->mt[kk] & 0x80000000U) | (rng->mt[kk+1] & 0x7fffffffU);
      rng->mt[kk] = rng->mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1U];
    }
    uint32_t y = (rng->mt[623] & 0x80000000U) | (rng->mt[0] & 0x7fffffffU);
    rng->mt[623] = rng->mt[396] ^ (y >> 1) ^ mag01[y & 1U];
    rng->index = 0;
  }
  uint32_t y = rng->mt[rng->index++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680U;
  y ^= (y << 15) & 0xefc60000U;
  y ^= (y >> 18);
  return y;
}

static inline float mt_uniform(Mt19937 *rng, float low, float high) {
  const double a = (double)(mt_extract(rng) >> 5);
  const double b = (double)(mt_extract(rng) >> 6);
  const double random = (a * 67108864.0 + b) / 9007199254740992.0;
  return (float)(low + (high - low) * random);
}

typedef struct {
  const char *name;
  int rows;
  int cols;
} DivTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
  int a_broadcast_cols;
  int b_broadcast_cols;
} CmpltTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} CmpeqTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} MinusTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} NegTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} AbsTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} AddTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} MulTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} WhereTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} RounddownTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} RoundoffTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} MaxpoolTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} AvgpoolTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} MaxTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} ReluTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} SiluTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} SigmoidTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} SinTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} TanTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} CosTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} AsinTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} AcosTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} AtanTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} AsinhTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} AcoshTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} TanhTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} SinhTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} CoshTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} AtanhTestConfig;

typedef struct {
  const char *name;
  int rows;
  int cols;
} LutTestConfig;

typedef struct {
  const char *name;
  int M;
  int K;
  int N;
} MatmulTestConfig;

typedef struct {
  const char *name;
  int batch;
  int in_channels;
  int input_size;
  int out_channels;
  int weight_in_channels;
  int kernel_size;
  int groups;
} Conv1dTestConfig;

typedef struct {
  int batch;
  int in_channels;
  int in_height;
  int in_width;
  int out_channels;
  int weight_in_channels;
  int kernel_h;
  int kernel_w;
  int groups;
  const char *name;
} Conv2dTestConfig;

typedef float (*LutRefFn)(float);

#endif
