// src/tools/gyrograph/gyrograph_opencl.c
//
// OpenCL trace backend for GyroGraph.
//
// This backend computes the 4-step Ω trace of many cells in parallel:
//   input:  omega12_in[n], words4_in[n,4]
//   output: omega_trace4_out[n,4], chi_trace4_out[n,4]
//
// Histograms / ring updates remain on CPU side, while the Ω stepping
// work is offloaded to OpenCL.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#define GYROGRAPH_EXPORT __declspec(dllexport)
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#else
#define GYROGRAPH_EXPORT __attribute__((visibility("default")))
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

static const char* GYROGRAPH_KERNEL_SOURCE =
"__kernel void gyrograph_trace_word4_batch(\n"
"    __global const long* cell_ids,\n"
"    __global const int* omega12_in,\n"
"    __global const uchar* words4_in,\n"
"    const int n,\n"
"    __global int* omega_trace4_out,\n"
"    __global uchar* chi_trace4_out\n"
") {\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= n) return;\n"
"\n"
"    int cid = (int)cell_ids[gid];\n"
"    uint s = ((uint)omega12_in[cid]) & 0x0FFFu;\n"
"    size_t base = ((size_t)gid) * 4u;\n"
"\n"
"    for (int k = 0; k < 4; ++k) {\n"
"        uchar b = words4_in[base + (size_t)k];\n"
"        uchar intron = b ^ (uchar)0xAA;\n"
"        uchar micro = (uchar)((intron >> 1) & 0x3F);\n"
"        uchar eps_a = (uchar)((intron & 0x01) ? 0x3F : 0x00);\n"
"        uchar eps_b = (uchar)((intron & 0x80) ? 0x3F : 0x00);\n"
"\n"
"        uint u6 = (s >> 6) & 0x3Fu;\n"
"        uint v6 = s & 0x3Fu;\n"
"        uint u_next = (v6 ^ eps_a) & 0x3Fu;\n"
"        uint v_next = (u6 ^ micro ^ eps_b) & 0x3Fu;\n"
"        s = (u_next << 6) | v_next;\n"
"\n"
"        omega_trace4_out[base + (size_t)k] = (int)s;\n"
"        chi_trace4_out[base + (size_t)k] = (uchar)(((s >> 6) ^ s) & 0x3F);\n"
"    }\n"
"}\n";

typedef struct {
    int initialized;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    cl_mem cell_ids_buf;
    cl_mem omega_in_buf;
    cl_mem words_buf;
    cl_mem omega_trace_buf;
    cl_mem chi_trace_buf;

    size_t cell_ids_capacity;
    size_t omega_in_capacity;
    size_t words_capacity;
    size_t omega_trace_capacity;
    size_t chi_trace_capacity;
} gyrograph_cl_runtime_t;

static gyrograph_cl_runtime_t G_RUNTIME = {0};

static void _release_buffer(cl_mem* buf, size_t* cap) {
    if (*buf) {
        clReleaseMemObject(*buf);
        *buf = 0;
    }
    if (cap) {
        *cap = 0;
    }
}

GYROGRAPH_EXPORT int gyrograph_cl_available(void) {
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
        if (err == CL_SUCCESS && ndevices > 0) {
            found = 1;
        }
    }

    free(platforms);
    return found ? 1 : 0;
}

GYROGRAPH_EXPORT void gyrograph_cl_shutdown(void) {
    if (!G_RUNTIME.initialized) return;

    _release_buffer(&G_RUNTIME.cell_ids_buf, &G_RUNTIME.cell_ids_capacity);
    _release_buffer(&G_RUNTIME.omega_in_buf, &G_RUNTIME.omega_in_capacity);
    _release_buffer(&G_RUNTIME.words_buf, &G_RUNTIME.words_capacity);
    _release_buffer(&G_RUNTIME.omega_trace_buf, &G_RUNTIME.omega_trace_capacity);
    _release_buffer(&G_RUNTIME.chi_trace_buf, &G_RUNTIME.chi_trace_capacity);

    if (G_RUNTIME.kernel) {
        clReleaseKernel(G_RUNTIME.kernel);
        G_RUNTIME.kernel = 0;
    }
    if (G_RUNTIME.program) {
        clReleaseProgram(G_RUNTIME.program);
        G_RUNTIME.program = 0;
    }
    if (G_RUNTIME.queue) {
        clReleaseCommandQueue(G_RUNTIME.queue);
        G_RUNTIME.queue = 0;
    }
    if (G_RUNTIME.context) {
        clReleaseContext(G_RUNTIME.context);
        G_RUNTIME.context = 0;
    }

    G_RUNTIME.initialized = 0;
}

GYROGRAPH_EXPORT int gyrograph_cl_init(int platform_index, int device_index) {
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
        gyrograph_cl_shutdown();
        return 0;
    }

    size_t src_len = strlen(GYROGRAPH_KERNEL_SOURCE);
    G_RUNTIME.program = clCreateProgramWithSource(
        G_RUNTIME.context,
        1,
        &GYROGRAPH_KERNEL_SOURCE,
        &src_len,
        &err
    );
    if (err != CL_SUCCESS || !G_RUNTIME.program) {
        gyrograph_cl_shutdown();
        return 0;
    }

    err = clBuildProgram(G_RUNTIME.program, 1, &G_RUNTIME.device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        gyrograph_cl_shutdown();
        return 0;
    }

    G_RUNTIME.kernel = clCreateKernel(
        G_RUNTIME.program,
        "gyrograph_trace_word4_batch",
        &err
    );
    if (err != CL_SUCCESS || !G_RUNTIME.kernel) {
        gyrograph_cl_shutdown();
        return 0;
    }

    G_RUNTIME.initialized = 1;
    return 1;
}

static int _ensure_buffer(cl_mem* buf, size_t* capacity, size_t bytes, cl_mem_flags flags) {
    cl_int err;
    if (*buf && *capacity >= bytes) {
        return 1;
    }

    _release_buffer(buf, capacity);

    *buf = clCreateBuffer(G_RUNTIME.context, flags, bytes, NULL, &err);
    if (err != CL_SUCCESS || !*buf) {
        return 0;
    }
    *capacity = bytes;
    return 1;
}

GYROGRAPH_EXPORT int gyrograph_cl_trace_word4_batch(
    const int64_t* cell_ids,
    const int32_t* omega12_in,
    const uint8_t* words4_in,
    int64_t n,
    int32_t* omega_trace4_out,
    uint8_t* chi_trace4_out
) {
    if (!G_RUNTIME.initialized) return 0;
    if (cell_ids == NULL || omega12_in == NULL || words4_in == NULL || omega_trace4_out == NULL || chi_trace4_out == NULL || n < 0) {
        return 0;
    }

    size_t cell_ids_bytes = (size_t)n * sizeof(int64_t);
    size_t omega_in_bytes = (size_t)n * sizeof(int32_t);
    size_t words_bytes = (size_t)n * 4u * sizeof(uint8_t);
    size_t omega_trace_bytes = (size_t)n * 4u * sizeof(int32_t);
    size_t chi_trace_bytes = (size_t)n * 4u * sizeof(uint8_t);

    if (!_ensure_buffer(&G_RUNTIME.cell_ids_buf, &G_RUNTIME.cell_ids_capacity, cell_ids_bytes, CL_MEM_READ_ONLY)) return 0;
    if (!_ensure_buffer(&G_RUNTIME.omega_in_buf, &G_RUNTIME.omega_in_capacity, omega_in_bytes, CL_MEM_READ_ONLY)) return 0;
    if (!_ensure_buffer(&G_RUNTIME.words_buf, &G_RUNTIME.words_capacity, words_bytes, CL_MEM_READ_ONLY)) return 0;
    if (!_ensure_buffer(&G_RUNTIME.omega_trace_buf, &G_RUNTIME.omega_trace_capacity, omega_trace_bytes, CL_MEM_WRITE_ONLY)) return 0;
    if (!_ensure_buffer(&G_RUNTIME.chi_trace_buf, &G_RUNTIME.chi_trace_capacity, chi_trace_bytes, CL_MEM_WRITE_ONLY)) return 0;

    cl_int err;
    err = clEnqueueWriteBuffer(G_RUNTIME.queue, G_RUNTIME.cell_ids_buf, CL_TRUE, 0, cell_ids_bytes, cell_ids, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    err = clEnqueueWriteBuffer(G_RUNTIME.queue, G_RUNTIME.omega_in_buf, CL_TRUE, 0, omega_in_bytes, omega12_in, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    err = clEnqueueWriteBuffer(G_RUNTIME.queue, G_RUNTIME.words_buf, CL_TRUE, 0, words_bytes, words4_in, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    err = clSetKernelArg(G_RUNTIME.kernel, 0, sizeof(cl_mem), &G_RUNTIME.cell_ids_buf);
    if (err != CL_SUCCESS) return 0;
    err = clSetKernelArg(G_RUNTIME.kernel, 1, sizeof(cl_mem), &G_RUNTIME.omega_in_buf);
    if (err != CL_SUCCESS) return 0;
    err = clSetKernelArg(G_RUNTIME.kernel, 2, sizeof(cl_mem), &G_RUNTIME.words_buf);
    if (err != CL_SUCCESS) return 0;
    err = clSetKernelArg(G_RUNTIME.kernel, 3, sizeof(int), (int[]){ (int)n });
    if (err != CL_SUCCESS) return 0;
    err = clSetKernelArg(G_RUNTIME.kernel, 4, sizeof(cl_mem), &G_RUNTIME.omega_trace_buf);
    if (err != CL_SUCCESS) return 0;
    err = clSetKernelArg(G_RUNTIME.kernel, 5, sizeof(cl_mem), &G_RUNTIME.chi_trace_buf);
    if (err != CL_SUCCESS) return 0;

    size_t global = (size_t)n;
    err = clEnqueueNDRangeKernel(G_RUNTIME.queue, G_RUNTIME.kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    err = clEnqueueReadBuffer(G_RUNTIME.queue, G_RUNTIME.omega_trace_buf, CL_TRUE, 0, omega_trace_bytes, omega_trace4_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    err = clEnqueueReadBuffer(G_RUNTIME.queue, G_RUNTIME.chi_trace_buf, CL_TRUE, 0, chi_trace_bytes, chi_trace4_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    return 1;
}