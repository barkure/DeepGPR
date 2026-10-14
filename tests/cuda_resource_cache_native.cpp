/* Native resource ownership checks. Build with a CUDA toolkit (no kernels):
 * c++ -std=c++14 -pthread -I$CUDA_HOME/include tests/cuda_resource_cache_native.cpp \
 *     -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64 -lcudart -o /tmp/cache-test
 * /tmp/cache-test
 */
#include "../src/DeepGPR/lib/cuda_resources.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>

static void checked(cudaError_t error)
{
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::abort();
    }
}

int main()
{
    checked(cudaSetDevice(0));
    CudaAsyncCache cache;
    cudaStream_t stream;
    unsigned char* pointer;
    {
        CudaAsyncWorkspace first;
        cache.take(0, first);
        checked(first.prepare(1024, 512));
        stream = first.streams[0];
        pointer = first.e;
        checked(cudaMemsetAsync(first.e, 7, 1024, stream));
        checked(cudaStreamSynchronize(stream));
        cache.put(first);
        assert(first.device == -1 && first.e == nullptr);
    }
    {
        CudaAsyncWorkspace next;
        cache.take(0, next);
        checked(next.prepare(512, 0));
        assert(next.e == pointer && next.streams[0] == stream);
        assert(next.e_capacity == 1024 && next.r_capacity == 512);
        checked(next.prepare(8192, 4096));
        assert(next.e_capacity == 8192 && next.r_capacity == 4096);
        cache.put(next);
    }
    cache.clear(0);
    {
        CudaAsyncWorkspace empty;
        cache.take(0, empty);
        assert(!empty.e && !empty.streams[0]);
        checked(empty.prepare(CudaAsyncCache::max_bytes + 1, 0));
        cache.put(empty);
        assert(empty.e != nullptr);  // oversized allocation was not retained
    }
    {
        CudaAsyncWorkspace failed;
        cache.take(0, failed);
        assert(!failed.e);
        // Fail after streams, events and the first buffer have been created.
        assert(failed.prepare(1024, std::numeric_limits<size_t>::max()) != cudaSuccess);
        cudaGetLastError();
        // Destruction releases the partially prepared workspace.
    }
    std::atomic<int> ready{0};
    unsigned char* pointers[4] = {};
    std::thread workers[4];
    for (int i = 0; i < 4; ++i) {
        workers[i] = std::thread([&, i] {
            checked(cudaSetDevice(0));
            CudaAsyncWorkspace workspace;
            cache.take(0, workspace);
            checked(workspace.prepare(1024, 1024));
            pointers[i] = workspace.e;
            ++ready;
            while (ready.load() < 4) std::this_thread::yield();
            checked(cudaMemsetAsync(workspace.e, i, 1024, workspace.streams[0]));
            checked(cudaStreamSynchronize(workspace.streams[0]));
            cache.put(workspace);
        });
    }
    for (auto& worker : workers) worker.join();
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) assert(pointers[i] != pointers[j]);
    }
    {
        CudaAsyncWorkspace first, second;
        cache.take(0, first);
        cache.take(0, second);
        assert(first.e && !second.e);  // only one of the four returns is cached
    }
    int devices;
    checked(cudaGetDeviceCount(&devices));
    if (devices > 1) {
        checked(cudaSetDevice(1));
        CudaAsyncWorkspace other;
        cache.take(1, other);
        checked(other.prepare(1024, 0));
        cache.put(other);
        checked(cudaSetDevice(0));
        cache.clear(1);
        int current;
        checked(cudaGetDevice(&current));
        assert(current == 0);
    } else {
        std::puts("Multi-device check skipped (one GPU)");
    }
    std::puts("CUDA resource ownership checks passed");
}
