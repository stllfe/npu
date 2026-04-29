project('ops_reg', 'c')

incdir = include_directories('../include')
ops_reg_exe = executable('ops_reg', 'main.c',
  include_directories: incdir,
  c_args: ['-Wno-unused-variable', '-Wno-unused-function'],
  link_args: ['-ldrm', '-lm'],
)

matmul_cases = [
  ['33', '1', '33'],
  ['34', '1', '34'],
  ['65', '1', '33'],
  ['32', '32', '32'],
  ['64', '64', '64'],
  ['128', '128', '128'],
  ['256', '256', '256'],
  ['394', '394', '394'], 
  ['1', '8192', '8192'], 
  ['1', '8192', '8193'], 
  ['1', '8193', '8192'], 
  ['1', '8193', '8193'], 
  ['1', '32', '16'],
  ['1', '394', '394'],
  ['1', '768', '768'],
  ['1', '768', '2048'],
  ['1', '2048', '2048'],
  ['1', '4096', '4096'],
]

foreach dims : matmul_cases
 test('matmul @0@x@1@x@2@'.format(dims[0], dims[1], dims[2]),
   ops_reg_exe,
   is_parallel: false,
   suite: 'matmul',
   args: ['matmul', dims[0], dims[1], dims[2]])
endforeach

# foreach n : range(1, 8192, 10)
#   test('matmul @0@x1x@0@'.format(n),
#     ops_reg_exe,
#     is_parallel: false,
#     suite: 'matmul',
#     timeout: 180,
#     args: ['matmul', n.to_string(), '1', n.to_string()])
# endforeach

# llm_linear_cases = [
#   ['1', '1024', '2048', 'attn_q_proj'],
#   ['1', '1024', '1024', 'attn_k_proj'],
#   ['1', '1024', '1024', 'attn_v_proj'],
#   ['1', '2048', '1024', 'attn_o_proj'],
#   ['1', '1024', '3072', 'ffn_gate_proj'],
#   ['1', '1024', '3072', 'ffn_up_proj'],
#   ['1', '3072', '1024', 'ffn_down_proj'],
#   ['1', '1024', '151936', 'lm_head'], # >30s
# ]

# foreach dims : llm_linear_cases
#   test('matmul @0@'.format(dims[3]),
#     ops_reg_exe,
#     is_parallel: false,
#     suite: 'matmul',
#     timeout: 180,
#     args: ['matmul', dims[0], dims[1], dims[2]])
# endforeach


##################

conv2d_cases = [
  ['1', '3', '5', '7', '6', '3', '2', '1', '1', 'conv2d_i1357_w6321'],
  ['1', '3', '5', '7', '6', '3', '2', '3', '1', 'conv2d_i1357_w6323'],
  ['1', '3', '5', '7', '6', '3', '2', '5', '1', 'conv2d_i1357_w6325'],
  ['1', '3', '5', '7', '6', '3', '3', '1', '1', 'conv2d_i1357_w6331'],
  ['1', '3', '5', '7', '6', '3', '3', '3', '1', 'conv2d_i1357_w6333'],
  ['1', '3', '5', '7', '6', '1', '3', '3', '3', 'conv2d_i1357_w6133_g3'],
  ['1', '3', '5', '7', '6', '3', '3', '5', '1', 'conv2d_i1357_w6335'],
  ['1', '16', '18', '18', '16', '16', '3', '3', '1', 'conv2d_i1161818_w161633'], 
  ['1', '4', '9', '9', '4', '4', '1', '1', '1', 'conv2d_i1499_w4411'], 
  ['1', '16', '32', '32', '16', '16', '1', '1', '1', 'conv2d_i1163232_w161611'], # failed
  ['2', '4', '9', '9', '4', '4', '3', '3', '1', 'conv2d_i2499_w4433'], # failed
  ['1', '1', '5', '7', '6', '1', '3', '3', '1', 'conv2d_i1157_w6133'], # failed
  ['1', '4', '1', '1', '2', '2', '1', '1', '2', 'conv2d_i11411_w2211_g2'], # failed
  ['1', '32', '32', '32', '32', '1', '1', '1', '32', 'conv2d_i1323232_w32111_g32'], # failed
  ['4', '15', '5', '5', '35', '3', '3', '3', '5', 'conv2d_i41555_w35333_g5'], # failed
  ['2', '3', '11', '28', '3', '1', '3', '3', '3', 'conv2d_i231128_w3133_g3'], # failed
]

foreach dims : conv2d_cases
  test(dims[9],
    ops_reg_exe,
    is_parallel: false,
    suite: 'conv2d',
    args: ['conv2d', dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7], dims[8]])
endforeach

foreach n : range(2032, 3000, 1)
  test('conv2d_1x3_@0@x@0@_k1'.format(n),
    ops_reg_exe,
    is_parallel: false,
    suite: 'conv2d',
    timeout: 180,
    args: ['conv2d', '1', '3', n.to_string(), n.to_string(), '6', '3', '1', '1', '1'])
endforeach

#################
# run_target('test_matmul',
#   command: ['meson', 'test', '--maxfail=1', '-C', meson.project_build_root(), '--suite', 'matmul'],
# )

run_target('test_conv2d',
  command: ['meson', 'test', '--maxfail=1', '-C', meson.project_build_root(), '--suite', 'conv2d'],
)
