// csrc/gyrolabe_opencl.c
//
// OpenCL backend for GyroLabe packed tensor GEMM.
// Accelerates only: packed matrix x packed batch of vectors -> output batch.
// CPU keeps: Moments, signatures, q-map, control-plane. GPU does dense multiply.
//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#define GYRO_EXPORT __declspec(dllexport)
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#else
#define GYRO_EXPORT __attribute__((visibility("default")))
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define GYRO_CL_MAX_MATRICES 32
#define GYRO_CL_MAX_HANDLE_ID 0x7FFFFFFF

GYRO_EXPORT void gyro_cl_shutdown(void);

static const char* GYRO_KERNEL_SOURCE =
    "#define N_BITS %d\n"
    "__kernel void gyro_gemm_packed_x_batch_f32_%d(\n"
    "    __global const ulong* W_sign,\n"
    "    __global const ulong* W_bp,\n"
    "    const float scale_w,\n"
    "    __global const float* scale_x,\n"
    "    __global const ulong* X_sign,\n"
    "    __global const ulong* X_bp,\n"
    "    const int rows,\n"
    "    const int cols,\n"
    "    const int batch,\n"
    "    __global float* Y\n"
    ") {\n"
    "    int row = get_global_id(0);\n"
    "    int b = get_global_id(1);\n"
    "    if (row >= rows || b >= batch) return;\n"
    "\n"
    "    ulong col_mask = (cols == 64) ? (ulong)(-1) : (((ulong)1 << cols) - 1);\n"
    "    ulong neg_mask = W_sign[row] ^ X_sign[b];\n"
    "    ulong pos_mask = (~neg_mask) & col_mask;\n"
    "\n"
    "    long pos_dot = 0;\n"
    "    long neg_dot = 0;\n"
    "\n"
    "    __global const ulong* wrow = W_bp + row * N_BITS;\n"
    "    __global const ulong* xvec = X_bp + b * N_BITS;\n"
    "\n"
    "    for (int m = 0; m < N_BITS; ++m) {\n"
    "        ulong wm = wrow[m];\n"
    "        for (int k = 0; k < N_BITS; ++k) {\n"
    "            ulong pm = wm & xvec[k];\n"
    "            long partial_pos = (long)popcount(pm & pos_mask);\n"
    "            long partial_neg = (long)popcount(pm & neg_mask);\n"
    "            int shift = m + k;\n"
    "            long mul = ((long)1) << shift;\n"
    "            pos_dot += partial_pos * mul;\n"
    "            neg_dot += partial_neg * mul;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    float scale_prod = scale_w * scale_x[b];\n"
    "    Y[b * rows + row] = ((float)(pos_dot - neg_dot)) / scale_prod;\n"
    "}\n"
    "__kernel void gyro_gemm64x64_packed_x_batch_f32_%d(\n"
    "    __global const ulong* W_sign,\n"
    "    __global const ulong* W_bp,\n"
    "    const float scale_w,\n"
    "    __global const float* scale_x,\n"
    "    __global const ulong* X_sign,\n"
    "    __global const ulong* X_bp,\n"
    "    const int rows,\n"
    "    const int cols,\n"
    "    const int batch,\n"
    "    __global float* Y\n"
    ") {\n"
    "    int row = get_global_id(0);\n"
    "    int b = get_global_id(1);\n"
    "    if (row >= 64 || b >= batch) return;\n"
    "\n"
    "    __local ulong lx_bp[N_BITS];\n"
    "    __local ulong lx_sign_arr[1];\n"
    "\n"
    "    if (get_local_id(0) == 0) {\n"
    "        lx_sign_arr[0] = X_sign[b];\n"
    "        __global const ulong* xvec_src = X_bp + ((size_t)b * (size_t)N_BITS);\n"
    "        for (int k = 0; k < N_BITS; ++k) {\n"
    "            lx_bp[k] = xvec_src[k];\n"
    "        }\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "\n"
    "    ulong neg_mask = W_sign[row] ^ lx_sign_arr[0];\n"
    "    ulong pos_mask = ~neg_mask;\n"
    "\n"
    "    long pos_dot = 0;\n"
    "    long neg_dot = 0;\n"
    "\n"
    "    __global const ulong* wrow = W_bp + ((size_t)row * (size_t)N_BITS);\n"
    "\n"
    "    #pragma unroll\n"
    "    for (int m = 0; m < N_BITS; ++m) {\n"
    "        ulong wm = wrow[m];\n"
    "        #pragma unroll\n"
    "        for (int k = 0; k < N_BITS; ++k) {\n"
    "            ulong pm = wm & lx_bp[k];\n"
    "            long partial_pos = (long)popcount(pm & pos_mask);\n"
    "            long partial_neg = (long)popcount(pm & neg_mask);\n"
    "            int shift = m + k;\n"
    "            long mul = ((long)1) << shift;\n"
    "            pos_dot += partial_pos * mul;\n"
    "            neg_dot += partial_neg * mul;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    float scale_prod = scale_w * scale_x[b];\n"
    "    Y[b * 64 + row] = ((float)(pos_dot - neg_dot)) / scale_prod;\n"
    "}\n";

static const char* GYRO_KERNEL_SOURCE_I32 =
    "#define N_BITS %d\n"
    "__kernel void gyro_gemm_packed_x_batch_i32_%d(\n"
    "    __global const ulong* W_sign,\n"
    "    __global const ulong* W_bp,\n"
    "    __global const ulong* X_sign,\n"
    "    __global const ulong* X_bp,\n"
    "    const int rows,\n"
    "    const int cols,\n"
    "    const int batch,\n"
    "    __global long* Y\n"
    ") {\n"
    "    int row = get_global_id(0);\n"
    "    int b = get_global_id(1);\n"
    "    if (row >= rows || b >= batch) return;\n"
    "\n"
    "    ulong col_mask = (cols == 64) ? (ulong)(-1) : (((ulong)1 << cols) - 1);\n"
    "    ulong neg_mask = W_sign[row] ^ X_sign[b];\n"
    "    ulong pos_mask = (~neg_mask) & col_mask;\n"
    "\n"
    "    long pos_dot = 0;\n"
    "    long neg_dot = 0;\n"
    "\n"
    "    __global const ulong* wrow = W_bp + row * N_BITS;\n"
    "    __global const ulong* xvec = X_bp + b * N_BITS;\n"
    "\n"
    "    for (int m = 0; m < N_BITS; ++m) {\n"
    "        ulong wm = wrow[m];\n"
    "        for (int k = 0; k < N_BITS; ++k) {\n"
    "            ulong pm = wm & xvec[k];\n"
    "            long partial_pos = (long)popcount(pm & pos_mask);\n"
    "            long partial_neg = (long)popcount(pm & neg_mask);\n"
    "            int shift = m + k;\n"
    "            long mul = ((long)1) << shift;\n"
    "            pos_dot += partial_pos * mul;\n"
    "            neg_dot += partial_neg * mul;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    Y[b * rows + row] = (long)(pos_dot - neg_dot);\n"
    "}\n"
    "__kernel void gyro_gemm64x64_packed_x_batch_i32_%d(\n"
    "    __global const ulong* W_sign,\n"
    "    __global const ulong* W_bp,\n"
    "    __global const ulong* X_sign,\n"
    "    __global const ulong* X_bp,\n"
    "    const int rows,\n"
    "    const int cols,\n"
    "    const int batch,\n"
    "    __global long* Y\n"
    ") {\n"
    "    int row = get_global_id(0);\n"
    "    int b = get_global_id(1);\n"
    "    if (row >= 64 || b >= batch) return;\n"
    "\n"
    "    __local ulong lx_bp[N_BITS];\n"
    "    __local ulong lx_sign_arr[1];\n"
    "\n"
    "    if (get_local_id(0) == 0) {\n"
    "        lx_sign_arr[0] = X_sign[b];\n"
    "        __global const ulong* xvec_src = X_bp + ((size_t)b * (size_t)N_BITS);\n"
    "        for (int k = 0; k < N_BITS; ++k) {\n"
    "            lx_bp[k] = xvec_src[k];\n"
    "        }\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "\n"
    "    ulong neg_mask = W_sign[row] ^ lx_sign_arr[0];\n"
    "    ulong pos_mask = ~neg_mask;\n"
    "\n"
    "    long pos_dot = 0;\n"
    "    long neg_dot = 0;\n"
    "\n"
    "    __global const ulong* wrow = W_bp + ((size_t)row * (size_t)N_BITS);\n"
    "\n"
    "    #pragma unroll\n"
    "    for (int m = 0; m < N_BITS; ++m) {\n"
    "        ulong wm = wrow[m];\n"
    "        #pragma unroll\n"
    "        for (int k = 0; k < N_BITS; ++k) {\n"
    "            ulong pm = wm & lx_bp[k];\n"
    "            long partial_pos = (long)popcount(pm & pos_mask);\n"
    "            long partial_neg = (long)popcount(pm & neg_mask);\n"
    "            int shift = m + k;\n"
    "            long mul = ((long)1) << shift;\n"
    "            pos_dot += partial_pos * mul;\n"
    "            neg_dot += partial_neg * mul;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    Y[b * 64 + row] = (long)(pos_dot - neg_dot);\n"
    "}\n";

typedef struct {
    uint64_t id;
    int64_t rows;
    int64_t cols;
    int32_t n_bits;
    float scale_w;
    cl_mem W_sign_buf;
    cl_mem W_bp_buf;
} gyro_cl_matrix_t;

typedef struct {
    int initialized;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    /* Persistent transient buffers reused across GEMM calls */
    cl_mem scale_buf;
    cl_mem xsign_buf;
    cl_mem xbp_buf;
    cl_mem y_buf;
    size_t scale_capacity_bytes;
    size_t xsign_capacity_bytes;
    size_t xbp_capacity_bytes;
    size_t y_capacity_bytes;
    cl_program program8;
    cl_program program12;
    cl_program program16;
    cl_kernel kernel8;
    cl_kernel kernel12;
    cl_kernel kernel16;
    cl_kernel kernel8_64;
    cl_kernel kernel12_64;
    cl_kernel kernel16_64;
    cl_program program8_i32;
    cl_program program12_i32;
    cl_program program16_i32;
    cl_kernel kernel8_i32;
    cl_kernel kernel12_i32;
    cl_kernel kernel16_i32;
    cl_kernel kernel8_i32_64;
    cl_kernel kernel12_i32_64;
    cl_kernel kernel16_i32_64;
    cl_mem y_i32_buf;
    size_t y_i32_capacity_bytes;
} gyro_cl_runtime_t;

static gyro_cl_runtime_t G_RUNTIME = {0};
static gyro_cl_matrix_t G_MATRICES[GYRO_CL_MAX_MATRICES];
static uint32_t G_NEXT_HANDLE = 1;

static void _build_program(cl_program* prog, cl_kernel* kern, cl_kernel* kern64,
                          int n_bits, cl_context ctx, cl_device_id dev, cl_int* err) {
    size_t src_len = 4096;
    char* src = (char*)malloc(src_len);
    if (!src) {
        *err = CL_OUT_OF_HOST_MEMORY;
        return;
    }
    int n = snprintf(src, src_len, GYRO_KERNEL_SOURCE, n_bits, n_bits, n_bits);
    if (n < 0 || (size_t)n >= src_len) {
        free(src);
        *err = CL_OUT_OF_HOST_MEMORY;
        return;
    }
    size_t actual_len = (size_t)n + 1;
    *err = CL_SUCCESS;
    *prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &actual_len, err);
    free(src);
    if (*err != CL_SUCCESS) return;
    *err = clBuildProgram(*prog, 1, &dev, NULL, NULL, NULL);
    if (*err != CL_SUCCESS) {
        clReleaseProgram(*prog);
        *prog = 0;
        return;
    }
    char name[64];
    snprintf(name, sizeof(name), "gyro_gemm_packed_x_batch_f32_%d", n_bits);
    *kern = clCreateKernel(*prog, name, err);
    if (*err != CL_SUCCESS) {
        clReleaseProgram(*prog);
        *prog = 0;
        return;
    }
    snprintf(name, sizeof(name), "gyro_gemm64x64_packed_x_batch_f32_%d", n_bits);
    *kern64 = clCreateKernel(*prog, name, err);
    if (*err != CL_SUCCESS) {
        *kern64 = 0;
        *err = CL_SUCCESS;
    }
}

static void _build_program_i32(cl_program* prog, cl_kernel* kern, cl_kernel* kern64,
                              int n_bits, cl_context ctx, cl_device_id dev, cl_int* err) {
    size_t src_len = 4096;
    char* src = (char*)malloc(src_len);
    if (!src) {
        *err = CL_OUT_OF_HOST_MEMORY;
        return;
    }
    int n = snprintf(src, src_len, GYRO_KERNEL_SOURCE_I32, n_bits, n_bits, n_bits);
    if (n < 0 || (size_t)n >= src_len) {
        free(src);
        *err = CL_OUT_OF_HOST_MEMORY;
        return;
    }
    size_t actual_len = (size_t)n + 1;
    *err = CL_SUCCESS;
    *prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &actual_len, err);
    free(src);
    if (*err != CL_SUCCESS) return;
    *err = clBuildProgram(*prog, 1, &dev, NULL, NULL, NULL);
    if (*err != CL_SUCCESS) {
        clReleaseProgram(*prog);
        *prog = 0;
        return;
    }
    char name[64];
    snprintf(name, sizeof(name), "gyro_gemm_packed_x_batch_i32_%d", n_bits);
    *kern = clCreateKernel(*prog, name, err);
    if (*err != CL_SUCCESS) {
        clReleaseProgram(*prog);
        *prog = 0;
        return;
    }
    snprintf(name, sizeof(name), "gyro_gemm64x64_packed_x_batch_i32_%d", n_bits);
    *kern64 = clCreateKernel(*prog, name, err);
    if (*err != CL_SUCCESS) {
        *kern64 = 0;
        *err = CL_SUCCESS;
    }
}

static void _release_matrix_slot(int idx) {
    gyro_cl_matrix_t* m = &G_MATRICES[idx];
    if (m->W_sign_buf) {
        clReleaseMemObject(m->W_sign_buf);
        m->W_sign_buf = 0;
    }
    if (m->W_bp_buf) {
        clReleaseMemObject(m->W_bp_buf);
        m->W_bp_buf = 0;
    }
    m->id = 0;
}

GYRO_EXPORT int gyro_cl_available(void) {
    cl_uint nplatforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &nplatforms);
    if (err != CL_SUCCESS || nplatforms == 0) return 0;
    cl_platform_id* platforms = (cl_platform_id*)malloc(nplatforms * sizeof(cl_platform_id));
    if (!platforms) return 0;
    err = clGetPlatformIDs(nplatforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        free(platforms);
        return 0;
    }
    int found = 0;
    for (cl_uint p = 0; p < nplatforms && !found; ++p) {
        cl_uint ndevices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &ndevices);
        if (err == CL_SUCCESS && ndevices > 0) found = 1;
    }
    free(platforms);
    return found ? 1 : 0;
}

GYRO_EXPORT int gyro_cl_init(int platform_index, int device_index) {
    if (G_RUNTIME.initialized) return 1;
    cl_int err;
    cl_uint nplatforms = 0;
    err = clGetPlatformIDs(0, NULL, &nplatforms);
    if (err != CL_SUCCESS || nplatforms == 0) return 0;
    if (platform_index < 0 || (cl_uint)platform_index >= nplatforms) return 0;
    cl_platform_id* platforms = (cl_platform_id*)malloc(nplatforms * sizeof(cl_platform_id));
    if (!platforms) return 0;
    err = clGetPlatformIDs(nplatforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        free(platforms);
        return 0;
    }
    G_RUNTIME.platform = platforms[platform_index];
    free(platforms);

    cl_uint ndevices = 0;
    err = clGetDeviceIDs(G_RUNTIME.platform, CL_DEVICE_TYPE_ALL, 0, NULL, &ndevices);
    if (err != CL_SUCCESS || ndevices == 0) return 0;
    if (device_index < 0 || (cl_uint)device_index >= ndevices) return 0;
    cl_device_id* devices = (cl_device_id*)malloc(ndevices * sizeof(cl_device_id));
    if (!devices) return 0;
    err = clGetDeviceIDs(G_RUNTIME.platform, CL_DEVICE_TYPE_ALL, ndevices, devices, NULL);
    if (err != CL_SUCCESS) {
        free(devices);
        return 0;
    }
    G_RUNTIME.device = devices[device_index];
    free(devices);

    G_RUNTIME.context = clCreateContext(NULL, 1, &G_RUNTIME.device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !G_RUNTIME.context) return 0;

    G_RUNTIME.queue = clCreateCommandQueue(G_RUNTIME.context, G_RUNTIME.device, 0, &err);
    if (err != CL_SUCCESS || !G_RUNTIME.queue) {
        clReleaseContext(G_RUNTIME.context);
        G_RUNTIME.context = 0;
        return 0;
    }

    _build_program(&G_RUNTIME.program8, &G_RUNTIME.kernel8, &G_RUNTIME.kernel8_64,
                  8, G_RUNTIME.context, G_RUNTIME.device, &err);
    if (err != CL_SUCCESS) {
        gyro_cl_shutdown();
        return 0;
    }
    _build_program(&G_RUNTIME.program12, &G_RUNTIME.kernel12, &G_RUNTIME.kernel12_64,
                  12, G_RUNTIME.context, G_RUNTIME.device, &err);
    if (err != CL_SUCCESS) {
        gyro_cl_shutdown();
        return 0;
    }
    _build_program(&G_RUNTIME.program16, &G_RUNTIME.kernel16, &G_RUNTIME.kernel16_64,
                  16, G_RUNTIME.context, G_RUNTIME.device, &err);
    if (err != CL_SUCCESS) {
        gyro_cl_shutdown();
        return 0;
    }

    _build_program_i32(&G_RUNTIME.program8_i32, &G_RUNTIME.kernel8_i32, &G_RUNTIME.kernel8_i32_64,
                       8, G_RUNTIME.context, G_RUNTIME.device, &err);
    if (err != CL_SUCCESS) {
        gyro_cl_shutdown();
        return 0;
    }
    _build_program_i32(&G_RUNTIME.program12_i32, &G_RUNTIME.kernel12_i32, &G_RUNTIME.kernel12_i32_64,
                       12, G_RUNTIME.context, G_RUNTIME.device, &err);
    if (err != CL_SUCCESS) {
        gyro_cl_shutdown();
        return 0;
    }
    _build_program_i32(&G_RUNTIME.program16_i32, &G_RUNTIME.kernel16_i32, &G_RUNTIME.kernel16_i32_64,
                       16, G_RUNTIME.context, G_RUNTIME.device, &err);
    if (err != CL_SUCCESS) {
        gyro_cl_shutdown();
        return 0;
    }

    G_RUNTIME.initialized = 1;
    return 1;
}

GYRO_EXPORT void gyro_cl_shutdown(void) {
    if (!G_RUNTIME.initialized) return;
    for (int i = 0; i < GYRO_CL_MAX_MATRICES; ++i) {
        _release_matrix_slot(i);
    }
    if (G_RUNTIME.scale_buf) {
        clReleaseMemObject(G_RUNTIME.scale_buf);
        G_RUNTIME.scale_buf = 0;
        G_RUNTIME.scale_capacity_bytes = 0;
    }
    if (G_RUNTIME.xsign_buf) {
        clReleaseMemObject(G_RUNTIME.xsign_buf);
        G_RUNTIME.xsign_buf = 0;
        G_RUNTIME.xsign_capacity_bytes = 0;
    }
    if (G_RUNTIME.xbp_buf) {
        clReleaseMemObject(G_RUNTIME.xbp_buf);
        G_RUNTIME.xbp_buf = 0;
        G_RUNTIME.xbp_capacity_bytes = 0;
    }
    if (G_RUNTIME.y_buf) {
        clReleaseMemObject(G_RUNTIME.y_buf);
        G_RUNTIME.y_buf = 0;
        G_RUNTIME.y_capacity_bytes = 0;
    }
    if (G_RUNTIME.y_i32_buf) {
        clReleaseMemObject(G_RUNTIME.y_i32_buf);
        G_RUNTIME.y_i32_buf = 0;
        G_RUNTIME.y_i32_capacity_bytes = 0;
    }
    if (G_RUNTIME.kernel8) { clReleaseKernel(G_RUNTIME.kernel8); G_RUNTIME.kernel8 = 0; }
    if (G_RUNTIME.kernel12) { clReleaseKernel(G_RUNTIME.kernel12); G_RUNTIME.kernel12 = 0; }
    if (G_RUNTIME.kernel16) { clReleaseKernel(G_RUNTIME.kernel16); G_RUNTIME.kernel16 = 0; }
    if (G_RUNTIME.kernel8_64) { clReleaseKernel(G_RUNTIME.kernel8_64); G_RUNTIME.kernel8_64 = 0; }
    if (G_RUNTIME.kernel12_64) { clReleaseKernel(G_RUNTIME.kernel12_64); G_RUNTIME.kernel12_64 = 0; }
    if (G_RUNTIME.kernel16_64) { clReleaseKernel(G_RUNTIME.kernel16_64); G_RUNTIME.kernel16_64 = 0; }
    if (G_RUNTIME.kernel8_i32) { clReleaseKernel(G_RUNTIME.kernel8_i32); G_RUNTIME.kernel8_i32 = 0; }
    if (G_RUNTIME.kernel12_i32) { clReleaseKernel(G_RUNTIME.kernel12_i32); G_RUNTIME.kernel12_i32 = 0; }
    if (G_RUNTIME.kernel16_i32) { clReleaseKernel(G_RUNTIME.kernel16_i32); G_RUNTIME.kernel16_i32 = 0; }
    if (G_RUNTIME.kernel8_i32_64) { clReleaseKernel(G_RUNTIME.kernel8_i32_64); G_RUNTIME.kernel8_i32_64 = 0; }
    if (G_RUNTIME.kernel12_i32_64) { clReleaseKernel(G_RUNTIME.kernel12_i32_64); G_RUNTIME.kernel12_i32_64 = 0; }
    if (G_RUNTIME.kernel16_i32_64) { clReleaseKernel(G_RUNTIME.kernel16_i32_64); G_RUNTIME.kernel16_i32_64 = 0; }
    if (G_RUNTIME.program8) { clReleaseProgram(G_RUNTIME.program8); G_RUNTIME.program8 = 0; }
    if (G_RUNTIME.program12) { clReleaseProgram(G_RUNTIME.program12); G_RUNTIME.program12 = 0; }
    if (G_RUNTIME.program16) { clReleaseProgram(G_RUNTIME.program16); G_RUNTIME.program16 = 0; }
    if (G_RUNTIME.program8_i32) { clReleaseProgram(G_RUNTIME.program8_i32); G_RUNTIME.program8_i32 = 0; }
    if (G_RUNTIME.program12_i32) { clReleaseProgram(G_RUNTIME.program12_i32); G_RUNTIME.program12_i32 = 0; }
    if (G_RUNTIME.program16_i32) { clReleaseProgram(G_RUNTIME.program16_i32); G_RUNTIME.program16_i32 = 0; }
    if (G_RUNTIME.queue) { clReleaseCommandQueue(G_RUNTIME.queue); G_RUNTIME.queue = 0; }
    if (G_RUNTIME.context) { clReleaseContext(G_RUNTIME.context); G_RUNTIME.context = 0; }
    G_RUNTIME.initialized = 0;
}

GYRO_EXPORT uint64_t gyro_cl_create_packed_matrix_f32(
    const uint64_t* W_sign,
    const uint64_t* W_bp,
    float scale_w,
    int64_t rows,
    int64_t cols,
    int32_t n_bits
) {
    if (!G_RUNTIME.initialized || !W_sign || !W_bp || rows < 0 || cols < 0 || cols > 64
        || n_bits != 8 && n_bits != 12 && n_bits != 16) {
        return 0;
    }
    int slot = -1;
    for (int i = 0; i < GYRO_CL_MAX_MATRICES; ++i) {
        if (G_MATRICES[i].id == 0) {
            slot = i;
            break;
        }
    }
    if (slot < 0) return 0;

    size_t sign_bytes = (size_t)rows * sizeof(uint64_t);
    size_t bp_bytes = (size_t)rows * (size_t)n_bits * sizeof(uint64_t);

    cl_int err;
    cl_mem sign_buf = clCreateBuffer(G_RUNTIME.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sign_bytes, (void*)W_sign, &err);
    if (err != CL_SUCCESS || !sign_buf) return 0;

    cl_mem bp_buf = clCreateBuffer(G_RUNTIME.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   bp_bytes, (void*)W_bp, &err);
    if (err != CL_SUCCESS || !bp_buf) {
        clReleaseMemObject(sign_buf);
        return 0;
    }

    uint64_t handle = (uint64_t)(G_NEXT_HANDLE++);
    if (G_NEXT_HANDLE > GYRO_CL_MAX_HANDLE_ID) G_NEXT_HANDLE = 1;

    gyro_cl_matrix_t* m = &G_MATRICES[slot];
    m->id = handle;
    m->rows = rows;
    m->cols = cols;
    m->n_bits = n_bits;
    m->scale_w = scale_w;
    m->W_sign_buf = sign_buf;
    m->W_bp_buf = bp_buf;
    return handle;
}

GYRO_EXPORT uint64_t gyro_cl_create_packed_matrix_i32(
    const uint64_t* W_sign,
    const uint64_t* W_bp,
    int64_t rows,
    int64_t cols,
    int32_t n_bits
) {
    return gyro_cl_create_packed_matrix_f32(W_sign, W_bp, 1.0f, rows, cols, n_bits);
}

GYRO_EXPORT void gyro_cl_release_packed_matrix(uint64_t handle) {
    if (handle == 0) return;
    for (int i = 0; i < GYRO_CL_MAX_MATRICES; ++i) {
        if (G_MATRICES[i].id == handle) {
            _release_matrix_slot(i);
            return;
        }
    }
}

static gyro_cl_matrix_t* _find_matrix(uint64_t handle) {
    if (handle == 0) return NULL;
    for (int i = 0; i < GYRO_CL_MAX_MATRICES; ++i) {
        if (G_MATRICES[i].id == handle) return &G_MATRICES[i];
    }
    return NULL;
}

GYRO_EXPORT int gyro_cl_gemm_packed_x_batch_f32(
    uint64_t handle,
    const float* scale_x,
    const uint64_t* X_sign,
    const uint64_t* X_bp,
    int64_t batch,
    float* Y_out
) {
    gyro_cl_matrix_t* m = _find_matrix(handle);
    if (!m || !G_RUNTIME.initialized || !scale_x || !X_sign || !X_bp || !Y_out || batch < 0) {
        return 0;
    }
    int64_t rows = m->rows;
    int32_t n_bits = m->n_bits;
    cl_kernel kern = NULL;
    if (rows == 64 && m->cols == 64) {
        if (n_bits == 8) kern = G_RUNTIME.kernel8_64 ? G_RUNTIME.kernel8_64 : G_RUNTIME.kernel8;
        else if (n_bits == 12) kern = G_RUNTIME.kernel12_64 ? G_RUNTIME.kernel12_64 : G_RUNTIME.kernel12;
        else if (n_bits == 16) kern = G_RUNTIME.kernel16_64 ? G_RUNTIME.kernel16_64 : G_RUNTIME.kernel16;
    } else {
        if (n_bits == 8) kern = G_RUNTIME.kernel8;
        else if (n_bits == 12) kern = G_RUNTIME.kernel12;
        else if (n_bits == 16) kern = G_RUNTIME.kernel16;
    }
    if (!kern) return 0;

    size_t scale_bytes = (size_t)batch * sizeof(float);
    size_t sign_bytes = (size_t)batch * sizeof(uint64_t);
    size_t bp_bytes = (size_t)batch * (size_t)n_bits * sizeof(uint64_t);
    size_t y_bytes = (size_t)batch * (size_t)rows * sizeof(float);

    cl_int err;

    if (!G_RUNTIME.scale_buf || G_RUNTIME.scale_capacity_bytes < scale_bytes) {
        if (G_RUNTIME.scale_buf) {
            clReleaseMemObject(G_RUNTIME.scale_buf);
            G_RUNTIME.scale_buf = 0;
            G_RUNTIME.scale_capacity_bytes = 0;
        }
        G_RUNTIME.scale_buf = clCreateBuffer(
            G_RUNTIME.context,
            CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
            scale_bytes,
            NULL,
            &err
        );
        if (err != CL_SUCCESS || !G_RUNTIME.scale_buf) return 0;
        G_RUNTIME.scale_capacity_bytes = scale_bytes;
    }

    if (!G_RUNTIME.xsign_buf || G_RUNTIME.xsign_capacity_bytes < sign_bytes) {
        if (G_RUNTIME.xsign_buf) {
            clReleaseMemObject(G_RUNTIME.xsign_buf);
            G_RUNTIME.xsign_buf = 0;
            G_RUNTIME.xsign_capacity_bytes = 0;
        }
        G_RUNTIME.xsign_buf = clCreateBuffer(
            G_RUNTIME.context,
            CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
            sign_bytes,
            NULL,
            &err
        );
        if (err != CL_SUCCESS || !G_RUNTIME.xsign_buf) return 0;
        G_RUNTIME.xsign_capacity_bytes = sign_bytes;
    }

    if (!G_RUNTIME.xbp_buf || G_RUNTIME.xbp_capacity_bytes < bp_bytes) {
        if (G_RUNTIME.xbp_buf) {
            clReleaseMemObject(G_RUNTIME.xbp_buf);
            G_RUNTIME.xbp_buf = 0;
            G_RUNTIME.xbp_capacity_bytes = 0;
        }
        G_RUNTIME.xbp_buf = clCreateBuffer(
            G_RUNTIME.context,
            CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
            bp_bytes,
            NULL,
            &err
        );
        if (err != CL_SUCCESS || !G_RUNTIME.xbp_buf) return 0;
        G_RUNTIME.xbp_capacity_bytes = bp_bytes;
    }

    if (!G_RUNTIME.y_buf || G_RUNTIME.y_capacity_bytes < y_bytes) {
        if (G_RUNTIME.y_buf) {
            clReleaseMemObject(G_RUNTIME.y_buf);
            G_RUNTIME.y_buf = 0;
            G_RUNTIME.y_capacity_bytes = 0;
        }
        G_RUNTIME.y_buf = clCreateBuffer(
            G_RUNTIME.context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            y_bytes,
            NULL,
            &err
        );
        if (err != CL_SUCCESS || !G_RUNTIME.y_buf) return 0;
        G_RUNTIME.y_capacity_bytes = y_bytes;
    }

    void* mapped = clEnqueueMapBuffer(
        G_RUNTIME.queue,
        G_RUNTIME.scale_buf,
        CL_TRUE,
        CL_MAP_WRITE,
        0,
        scale_bytes,
        0,
        NULL,
        NULL,
        &err
    );
    if (err != CL_SUCCESS || !mapped) return 0;
    memcpy(mapped, scale_x, scale_bytes);
    err = clEnqueueUnmapMemObject(G_RUNTIME.queue, G_RUNTIME.scale_buf, mapped, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    mapped = clEnqueueMapBuffer(
        G_RUNTIME.queue,
        G_RUNTIME.xsign_buf,
        CL_TRUE,
        CL_MAP_WRITE,
        0,
        sign_bytes,
        0,
        NULL,
        NULL,
        &err
    );
    if (err != CL_SUCCESS || !mapped) return 0;
    memcpy(mapped, X_sign, sign_bytes);
    err = clEnqueueUnmapMemObject(G_RUNTIME.queue, G_RUNTIME.xsign_buf, mapped, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    mapped = clEnqueueMapBuffer(
        G_RUNTIME.queue,
        G_RUNTIME.xbp_buf,
        CL_TRUE,
        CL_MAP_WRITE,
        0,
        bp_bytes,
        0,
        NULL,
        NULL,
        &err
    );
    if (err != CL_SUCCESS || !mapped) return 0;
    memcpy(mapped, X_bp, bp_bytes);
    err = clEnqueueUnmapMemObject(G_RUNTIME.queue, G_RUNTIME.xbp_buf, mapped, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    err = clSetKernelArg(kern, 0, sizeof(cl_mem), &m->W_sign_buf);
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 1, sizeof(cl_mem), &m->W_bp_buf);
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 2, sizeof(float), &m->scale_w);
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 3, sizeof(cl_mem), &G_RUNTIME.scale_buf);
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 4, sizeof(cl_mem), &G_RUNTIME.xsign_buf);
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 5, sizeof(cl_mem), &G_RUNTIME.xbp_buf);
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 6, sizeof(int), (int[]){ (int)rows });
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 7, sizeof(int), (int[]){ (int)m->cols });
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 8, sizeof(int), (int[]){ (int)batch });
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 9, sizeof(cl_mem), &G_RUNTIME.y_buf);
    if (err != CL_SUCCESS) goto cleanup;

    size_t global[2] = { (size_t)rows, (size_t)batch };
    size_t* local_ptr = NULL;
    size_t local[2];
    if (rows == 64 && m->cols == 64 && kern != NULL &&
        (kern == G_RUNTIME.kernel8_64 || kern == G_RUNTIME.kernel12_64 || kern == G_RUNTIME.kernel16_64)) {
        local[0] = 64;
        local[1] = 1;
        local_ptr = local;
    }
    err = clEnqueueNDRangeKernel(G_RUNTIME.queue, kern, 2, NULL, global, local_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto cleanup;

    mapped = clEnqueueMapBuffer(
        G_RUNTIME.queue,
        G_RUNTIME.y_buf,
        CL_TRUE,
        CL_MAP_READ,
        0,
        y_bytes,
        0,
        NULL,
        NULL,
        &err
    );
    if (err != CL_SUCCESS || !mapped) goto cleanup;
    memcpy(Y_out, mapped, y_bytes);
    err = clEnqueueUnmapMemObject(G_RUNTIME.queue, G_RUNTIME.y_buf, mapped, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto cleanup;

cleanup:
    return (err == CL_SUCCESS) ? 1 : 0;
}

GYRO_EXPORT int gyro_cl_gemm_packed_x_batch_i32(
    uint64_t handle,
    const uint64_t* X_sign,
    const uint64_t* X_bp,
    int64_t batch,
    int64_t* Y_out
) {
    gyro_cl_matrix_t* m = _find_matrix(handle);
    if (!m || !G_RUNTIME.initialized || !X_sign || !X_bp || !Y_out || batch < 0) {
        return 0;
    }
    int64_t rows = m->rows;
    int32_t n_bits = m->n_bits;
    cl_kernel kern = NULL;
    if (rows == 64 && m->cols == 64) {
        if (n_bits == 8) kern = G_RUNTIME.kernel8_i32_64 ? G_RUNTIME.kernel8_i32_64 : G_RUNTIME.kernel8_i32;
        else if (n_bits == 12) kern = G_RUNTIME.kernel12_i32_64 ? G_RUNTIME.kernel12_i32_64 : G_RUNTIME.kernel12_i32;
        else if (n_bits == 16) kern = G_RUNTIME.kernel16_i32_64 ? G_RUNTIME.kernel16_i32_64 : G_RUNTIME.kernel16_i32;
    } else {
        if (n_bits == 8) kern = G_RUNTIME.kernel8_i32;
        else if (n_bits == 12) kern = G_RUNTIME.kernel12_i32;
        else if (n_bits == 16) kern = G_RUNTIME.kernel16_i32;
    }
    if (!kern) return 0;

    size_t sign_bytes = (size_t)batch * sizeof(uint64_t);
    size_t bp_bytes = (size_t)batch * (size_t)n_bits * sizeof(uint64_t);
    size_t y_bytes = (size_t)batch * (size_t)rows * sizeof(int64_t);

    cl_int err;

    if (!G_RUNTIME.xsign_buf || G_RUNTIME.xsign_capacity_bytes < sign_bytes) {
        if (G_RUNTIME.xsign_buf) {
            clReleaseMemObject(G_RUNTIME.xsign_buf);
            G_RUNTIME.xsign_buf = 0;
            G_RUNTIME.xsign_capacity_bytes = 0;
        }
        G_RUNTIME.xsign_buf = clCreateBuffer(
            G_RUNTIME.context,
            CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
            sign_bytes,
            NULL,
            &err
        );
        if (err != CL_SUCCESS || !G_RUNTIME.xsign_buf) return 0;
        G_RUNTIME.xsign_capacity_bytes = sign_bytes;
    }

    if (!G_RUNTIME.xbp_buf || G_RUNTIME.xbp_capacity_bytes < bp_bytes) {
        if (G_RUNTIME.xbp_buf) {
            clReleaseMemObject(G_RUNTIME.xbp_buf);
            G_RUNTIME.xbp_buf = 0;
            G_RUNTIME.xbp_capacity_bytes = 0;
        }
        G_RUNTIME.xbp_buf = clCreateBuffer(
            G_RUNTIME.context,
            CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
            bp_bytes,
            NULL,
            &err
        );
        if (err != CL_SUCCESS || !G_RUNTIME.xbp_buf) return 0;
        G_RUNTIME.xbp_capacity_bytes = bp_bytes;
    }

    if (!G_RUNTIME.y_i32_buf || G_RUNTIME.y_i32_capacity_bytes < y_bytes) {
        if (G_RUNTIME.y_i32_buf) {
            clReleaseMemObject(G_RUNTIME.y_i32_buf);
            G_RUNTIME.y_i32_buf = 0;
            G_RUNTIME.y_i32_capacity_bytes = 0;
        }
        G_RUNTIME.y_i32_buf = clCreateBuffer(
            G_RUNTIME.context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            y_bytes,
            NULL,
            &err
        );
        if (err != CL_SUCCESS || !G_RUNTIME.y_i32_buf) return 0;
        G_RUNTIME.y_i32_capacity_bytes = y_bytes;
    }

    void* mapped = clEnqueueMapBuffer(
        G_RUNTIME.queue,
        G_RUNTIME.xsign_buf,
        CL_TRUE,
        CL_MAP_WRITE,
        0,
        sign_bytes,
        0,
        NULL,
        NULL,
        &err
    );
    if (err != CL_SUCCESS || !mapped) return 0;
    memcpy(mapped, X_sign, sign_bytes);
    err = clEnqueueUnmapMemObject(G_RUNTIME.queue, G_RUNTIME.xsign_buf, mapped, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    mapped = clEnqueueMapBuffer(
        G_RUNTIME.queue,
        G_RUNTIME.xbp_buf,
        CL_TRUE,
        CL_MAP_WRITE,
        0,
        bp_bytes,
        0,
        NULL,
        NULL,
        &err
    );
    if (err != CL_SUCCESS || !mapped) return 0;
    memcpy(mapped, X_bp, bp_bytes);
    err = clEnqueueUnmapMemObject(G_RUNTIME.queue, G_RUNTIME.xbp_buf, mapped, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    err = clSetKernelArg(kern, 0, sizeof(cl_mem), &m->W_sign_buf);
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 1, sizeof(cl_mem), &m->W_bp_buf);
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 2, sizeof(cl_mem), &G_RUNTIME.xsign_buf);
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 3, sizeof(cl_mem), &G_RUNTIME.xbp_buf);
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 4, sizeof(int), (int[]){ (int)rows });
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 5, sizeof(int), (int[]){ (int)m->cols });
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 6, sizeof(int), (int[]){ (int)batch });
    if (err != CL_SUCCESS) goto cleanup;
    err = clSetKernelArg(kern, 7, sizeof(cl_mem), &G_RUNTIME.y_i32_buf);
    if (err != CL_SUCCESS) goto cleanup;

    size_t global[2] = { (size_t)rows, (size_t)batch };
    size_t* local_ptr = NULL;
    size_t local[2];
    if (rows == 64 && m->cols == 64 && kern != NULL &&
        (kern == G_RUNTIME.kernel8_i32_64 || kern == G_RUNTIME.kernel12_i32_64 || kern == G_RUNTIME.kernel16_i32_64)) {
        local[0] = 64;
        local[1] = 1;
        local_ptr = local;
    }
    err = clEnqueueNDRangeKernel(G_RUNTIME.queue, kern, 2, NULL, global, local_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto cleanup;

    mapped = clEnqueueMapBuffer(
        G_RUNTIME.queue,
        G_RUNTIME.y_i32_buf,
        CL_TRUE,
        CL_MAP_READ,
        0,
        y_bytes,
        0,
        NULL,
        NULL,
        &err
    );
    if (err != CL_SUCCESS || !mapped) goto cleanup;
    memcpy(Y_out, mapped, y_bytes);
    err = clEnqueueUnmapMemObject(G_RUNTIME.queue, G_RUNTIME.y_i32_buf, mapped, 0, NULL, NULL);

cleanup:
    return (err == CL_SUCCESS) ? 1 : 0;
}
