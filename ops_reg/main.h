#ifndef OPS_REG_MAIN_H
#define OPS_REG_MAIN_H

#include <stdint.h>

typedef struct {
  uint32_t mt[624];
  int index;
} Mt19937;

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
  const char *fixture_dir;
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
