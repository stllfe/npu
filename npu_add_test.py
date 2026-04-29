#!/usr/bin/env python3
import numpy as np, ctypes, os, struct
from ctypes import byref, c_uint32, c_uint64, c_int, Structure, CDLL, c_void_p

DRM_RKNPU_MEM_CREATE = 0x40104401
DRM_RKNPU_MEM_MAP = 0x40104402
DRM_RKNPU_MEM_DESTROY = 0x40104403
DRM_RKNPU_MEM_SYNC = 0x40104404
DRM_RKNPU_SUBMIT = 0x40104a01
DRM_RKNPU_ACTION = 0x40104a02

RKNPU_MEM_NON_CACHEABLE = 2
RKNPU_MEM_SYNC_TO_DEVICE = 1
RKNPU_MEM_SYNC_FROM_DEVICE = 2
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC, RKNPU_JOB_BLOCK, RKNPU_JOB_PINGPONG = 1, 2, 8

DPU, DPU_RDMA = 0x1000, 0x2000
REG_DPU_FEATURE_MODE_CFG = 0x04000
REG_DPU_DATA_FORMAT = 0x04008
REG_DPU_DATA_CUBE_CHANNEL = 0x0400c
REG_DPU_DATA_CUBE_WIDTH = 0x04010
REG_DPU_EW_CFG = 0x04070
REG_DPU_DST_BASE_ADDR = 0x04030
REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH = 0x2050
REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT = 0x2054
REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL = 0x2058
REG_DPU_RDMA_RDMA_SRC_BASE_ADDR = 0x2060
REG_DPU_RDMA_RDMA_ERDMA_CFG = 0x207c
REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG = 0x2040

_libc = CDLL("libc.so.6", use_errno=True)
_ioctl = _libc.ioctl
_ioctl.argtypes = [c_int, c_uint64, c_void_p]
_ioctl.restype = c_int

def ioctl(fd, req, arg):
    r = _ioctl(fd, req, byref(arg))
    if r < 0: raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))

def reg(val, shift, mask): return ((val) << shift) & mask

def emit(q, target, r, v): q.append(((target+1)<<48) | ((v&0xffffffff)<<16) | (r&0xffff))

class MemAlloc:
    def __init__(self, fd, size):
        self.fd, self.size = fd, (size+0xfff)&~0xfff
        class S(Structure): _fields_ = [("flags",c_uint32),("size",c_uint32),("handle",c_uint32),("dma_addr",c_uint64)]
        s = S(flags=RKNPU_MEM_NON_CACHEABLE, size=self.size)
        ioctl(fd, DRM_RKNPU_MEM_CREATE, s)
        self.handle, self.dma = s.handle, s.dma_addr
        class M(Structure): _fields_ = [("handle",c_uint32),("offset",c_uint64)]
        ioctl(fd, DRM_RKNPU_MEM_MAP, M(handle=self.handle))
        self.ptr = (ctypes.c_uint8 * self.size)()
    
    def sync(self, f):
        class Sy(Structure): _fields_ = [("flags",c_uint32),("reserved",c_uint32),("obj_addr",c_uint64),("offset",c_uint64),("size",c_uint64)]
        ioctl(self.fd, DRM_RKNPU_MEM_SYNC, Sy(flags=f, obj_addr=self.dma, offset=0, size=self.size))
    
    def free(self):
        class D(Structure): _fields_ = [("handle",c_uint32),("reserved",c_uint32),("obj_addr",c_uint64)]
        ioctl(self.fd, DRM_RKNPU_MEM_DESTROY, D(handle=self.handle, obj_addr=self.dma))

class NPUEmulator:
    def __init__(self):
        self.hardware_ops = {'ADD': 2, 'MUL': 9, 'SUB': 4, 'MAX': 0, 'NEG': 19, 'FDIV': 3}
    
    def exec_alu(self, op, a, b):
        if op == 'ADD': return a + b
        if op == 'MUL': return a * b
        if op == 'SUB': return a - b
        if op == 'MAX': return np.maximum(a, b)
        if op == 'NEG': return -a
        if op == 'FDIV': return a / b
        raise NotImplementedError(f"op {op}")
    
    def run_add_real(self, fd, a_np, b_np):
        N = a_np.size
        ma, mb, mo, mt = [MemAlloc(fd, N*2) for _ in range(3)] + [MemAlloc(fd, 16384)]
        
        ctypes.memmove(ma.ptr, a_np.tobytes(), N*2)
        ctypes.memmove(mb.ptr, b_np.tobytes(), N*2)
        ma.sync(RKNPU_MEM_SYNC_TO_DEVICE); mb.sync(RKNPU_MEM_SYNC_TO_DEVICE)
        
        q = []
        emit(q, 0x11, REG_DPU_FEATURE_MODE_CFG, reg(15,0,0xf)|reg(2,8,0x300)|reg(1,12,0x1000))
        emit(q, 0x11, REG_DPU_DATA_FORMAT, reg(2,0,3)|reg(2,4,0x30)|reg(2,8,0x300))
        emit(q, 0x11, REG_DPU_DATA_CUBE_CHANNEL, reg(7,0,0xf)|reg(7,4,0xf0))
        emit(q, 0x11, REG_DPU_DATA_CUBE_WIDTH, reg(N-1,0,0x7ff))
        emit(q, 0x11, REG_DPU_EW_CFG, reg(0,31,0x80000000)|reg(1,28,0x30000000)|reg(2,22,0x00c00000)|reg(2,16,0x000f0000)|reg(1,12,0x00001000)|reg(1,8,0x00000100))
        emit(q, 0x21, 0x2050, reg(N-1,0,0x7ff)); emit(q, 0x21, 0x2054, 0)
        emit(q, 0x21, 0x2058, reg(7,0,0xf)); emit(q, 0x21, 0x2060, reg(ma.dma,0,0xffffffff))
        emit(q, 0x21, 0x207c, reg(1,0,1)|reg(2,4,0x30))
        emit(q, 0x21, 0x2040, reg(2,0,3)|reg(15,4,0xf0)|reg(2,8,0x300)|reg(1,12,0x1000))
        emit(q, 0x11, 0x4030, reg(mo.dma,0,0xffffffff))
        emit(q, 0x81, 8, 0x180008)
        
        for i,v in enumerate(q): ctypes.memmove(mt.ptr[i*8:], struct.pack("<Q",v), 8)
        
        class Task(Structure): _fields_ = [("flags",c_uint32),("op_idx",c_uint32),("enable_mask",c_uint32),("int_mask",c_uint32),("int_clear",c_uint32),("int_status",c_uint32),("regcfg_amount",c_uint32),("regcfg_offset",c_uint32),("regcmd_addr",c_uint64)]
        class Submit(Structure): _fields_ = [("flags",c_uint32),("timeout",c_uint32),("task_start",c_uint32),("task_number",c_uint32),("task_counter",c_uint32),("priority",c_uint32),("task_obj_addr",c_uint64),("reserved",c_uint32),("task_base_addr",c_uint64),("core_mask",c_uint32),("fence_fd",c_int)]
        class Act(Structure): _fields_ = [("flags",c_uint32)]
        
        ioctl(fd, DRM_RKNPU_ACTION, Act(flags=RKNPU_ACT_RESET))
        ioctl(fd, DRM_RKNPU_SUBMIT, Submit(flags=RKNPU_JOB_PC|RKNPU_JOB_BLOCK|RKNPU_JOB_PINGPONG, timeout=6000, task_start=0, task_number=1, task_counter=0, priority=0, task_obj_addr=mt.dma, reserved=0, task_base_addr=0, core_mask=1, fence_fd=-1))
        
        mo.sync(RKNPU_MEM_SYNC_FROM_DEVICE)
        out = np.frombuffer(bytes(mo.ptr[:N*2]), dtype=np.float16).copy()
        for m in [ma,mb,mo,mt]: m.free()
        return out
    
    def run_add(self, a, b):
        return self.exec_alu('ADD', a, b)

if __name__ == "__main__":
    npu = NPUEmulator()
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float16)
    
    print("=== Emulator (pure Python) ===")
    c = npu.run_add(a, b)
    print(f"A: {a}\nB: {b}\nA+B: {c}\nExpected: {a+b}")
    
    print("\n=== Real NPU (requires /dev/dri/card1) ===")
    try:
        fd = os.open("/dev/dri/card1", os.O_RDWR)
        c_real = npu.run_add_real(fd, a, b)
        print(f"A+B (NPU): {c_real}")
        os.close(fd)
    except Exception as e:
        print(f"Error: {e}")
        print("Run 'sudo modprobe rockchip-npu' first, or use emulator above")
