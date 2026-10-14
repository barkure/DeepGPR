#ifndef DEEPGPR_CUDA_RESOURCES_H
#define DEEPGPR_CUDA_RESOURCES_H

#include <cuda_runtime.h>
#include <list>
#include <mutex>
#include <new>
#include <utility>

/* A workspace is leased exclusively to one native call. Only drained,
 * successful calls return it to the cache; histories remain caller-owned. */
struct CudaAsyncWorkspace {
    int device = -1;
    unsigned char* e = nullptr;
    unsigned char* r = nullptr;
    size_t e_capacity = 0;
    size_t r_capacity = 0;
    cudaStream_t streams[2] = {nullptr, nullptr};
    cudaEvent_t events[4] = {nullptr, nullptr, nullptr, nullptr};

    CudaAsyncWorkspace() = default;
    CudaAsyncWorkspace(const CudaAsyncWorkspace&) = delete;
    CudaAsyncWorkspace& operator=(const CudaAsyncWorkspace&) = delete;

    void swap(CudaAsyncWorkspace& other) noexcept
    {
        std::swap(device, other.device);
        std::swap(e, other.e);
        std::swap(r, other.r);
        std::swap(e_capacity, other.e_capacity);
        std::swap(r_capacity, other.r_capacity);
        for (int i = 0; i < 2; ++i) std::swap(streams[i], other.streams[i]);
        for (int i = 0; i < 4; ++i) std::swap(events[i], other.events[i]);
    }

    ~CudaAsyncWorkspace()
    {
        if (device < 0) return;
        int previous;
        // During runtime shutdown CUDA may already be unavailable.
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
        for (auto stream : streams) {
            if (stream) cudaStreamSynchronize(stream);
        }
        if (e) cudaFree(e);
        if (r) cudaFree(r);
        for (auto event : events) {
            if (event) cudaEventDestroy(event);
        }
        for (auto stream : streams) {
            if (stream) cudaStreamDestroy(stream);
        }
        if (previous != device) cudaSetDevice(previous);
    }

    static cudaError_t reserve(unsigned char*& pointer, size_t& capacity, size_t bytes)
    {
        if (bytes <= capacity) return cudaSuccess;
        // The old workspace is idle. Release before growing to avoid holding
        // both old and new allocations at peak memory usage.
        if (pointer) {
            cudaError_t error = cudaFree(pointer);
            if (error != cudaSuccess) return error;
            pointer = nullptr;
            capacity = 0;
        }
        cudaError_t error = cudaMalloc(&pointer, bytes);
        if (error == cudaSuccess) capacity = bytes;
        return error;
    }

    cudaError_t prepare(size_t e_bytes, size_t r_bytes)
    {
        for (auto& stream : streams) {
            if (!stream) {
                cudaError_t error = cudaStreamCreate(&stream);
                if (error != cudaSuccess) return error;
            }
        }
        for (auto& event : events) {
            if (!event) {
                cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
                if (error != cudaSuccess) return error;
            }
        }
        cudaError_t error = reserve(e, e_capacity, e_bytes);
        if (error != cudaSuccess) return error;
        return reserve(r, r_capacity, r_bytes);
    }
};

class CudaAsyncCache {
    std::mutex mutex_;
    std::list<CudaAsyncWorkspace> idle_;

public:
    // Retain at most one idle workspace and 64 MiB of staging per device.
    // Oversized calls still run, but their workspace is released on return.
    static constexpr size_t max_bytes = 64 * size_t(1024) * 1024;

    void take(int device, CudaAsyncWorkspace& workspace)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = idle_.begin(); it != idle_.end(); ++it) {
            if (it->device == device) {
                workspace.swap(*it);
                idle_.erase(it);
                return;
            }
        }
        workspace.device = device;
    }

    void put(CudaAsyncWorkspace& workspace)
    {
        if (workspace.e_capacity > max_bytes
            || workspace.r_capacity > max_bytes - workspace.e_capacity) return;
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& entry : idle_) {
            if (entry.device == workspace.device) return;
        }
        // A cache miss must not turn a successful solver call into an error.
        try {
            idle_.emplace_back();
        } catch (const std::bad_alloc&) {
            return;
        }
        idle_.back().swap(workspace);
    }

    void clear(int device)
    {
        CudaAsyncWorkspace discarded;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = idle_.begin(); it != idle_.end(); ++it) {
                if (it->device == device) {
                    discarded.swap(*it);
                    idle_.erase(it);
                    break;
                }
            }
        }
        // CUDA calls happen outside the lock. Active calls own separate
        // workspaces and may repopulate the cache after clear() returns.
    }
};

inline CudaAsyncCache& cuda_async_cache()
{
    static CudaAsyncCache cache;
    return cache;
}

#endif
