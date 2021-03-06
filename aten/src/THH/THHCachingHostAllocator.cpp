// !!! This is a file automatically generated by hipify!!!
#include <THH/THHCachingHostAllocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/hip/detail/HIPHooks.h>


#include <hip/hip_runtime_api.h>
#include <deque>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <utility>

namespace {

struct BlockSize
{
  size_t  size; // allocation size
  void*   ptr;  // host memory pointer

  BlockSize(size_t size, void* ptr=NULL) : size(size), ptr(ptr) {}
};

struct Block : public BlockSize
{
  bool  allocated;    // true if the block is currently allocated
  int   event_count;  // number of outstanding cuda events
  std::unordered_set<at::hip::HIPStreamMasqueradingAsCUDA> streams;

  Block(size_t size, void* ptr, bool allocated) :
      BlockSize(size, ptr), allocated(allocated), event_count(0), streams() {}
};

static bool BlockComparator(const BlockSize& a, const BlockSize& b)
{
  // sort by size, break ties with pointer
  if (a.size != b.size) {
    return a.size < b.size;
  }
  return (uintptr_t)a.ptr < (uintptr_t)b.ptr;
}

struct HostAllocator
{
  typedef bool (*Comparison)(const BlockSize&, const BlockSize&);

  // lock around all operations
  std::mutex mutex;

  // blocks by pointer
  std::unordered_map<void*, Block> blocks;

  // pointers that are ready to be allocated (event_count=0)
  std::set<BlockSize, Comparison> available;

  // outstanding cuda events
  std::deque<std::pair<hipEvent_t, void*>> cuda_events;

  HostAllocator() : available(BlockComparator) {}

  hipError_t malloc(void** ptr, size_t size)
  {
    std::lock_guard<std::mutex> lock(mutex);

    // process outstanding cuda events which may have occurred
    hipError_t err = processEvents();
    if (err != hipSuccess) {
      return err;
    }

    // search for the smallest block which can hold this allocation
    BlockSize search_key(size);
    auto it = available.lower_bound(search_key);
    if (it != available.end()) {
      Block& block = blocks.at(it->ptr);
      THAssert(!block.allocated && block.event_count == 0);
      block.allocated = true;
      *ptr = block.ptr;
      available.erase(it);
      return hipSuccess;
    }

    // Pinned memory pointers allocated by any device can be directly used by any
    // other device, regardless of the current device at the time of allocation,
    // since we assume unified addressing.
    // So we grab any existing primary context, if available.
    // See pytorch/pytorch#21081.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index = at::cuda::detail::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
    }

    // note that hipHostMalloc may not touch pointer if size is 0
    *ptr = 0;

    // allocate a new block if no cached allocation is found
    err = hipHostMalloc(ptr, size, hipHostMallocDefault);
    if (err != hipSuccess) {
      return err;
    }

    blocks.insert({*ptr, Block(size, *ptr, true)});
    return hipSuccess;
  }

  hipError_t free(void* ptr)
  {
    std::lock_guard<std::mutex> lock(mutex);

    if (!ptr) {
      return hipSuccess;
    }

    // process outstanding cuda events which may have occurred
    hipError_t err = processEvents();
    if (err != hipSuccess) {
      return err;
    }

    auto it = blocks.find(ptr);
    THAssert(it != blocks.end());

    Block& block = it->second;
    THAssert(block.allocated);

    // free (on valid memory) shouldn't fail, so mark unallocated before
    // we process the streams.
    block.allocated = false;

    // insert CUDA events for each stream on which this block was used. This
    err = insertEvents(block);
    if (err != hipSuccess) {
      return err;
    }

    if (block.event_count == 0) {
      // the block can be re-used if there are no outstanding cuda events
      available.insert(block);
    }
    return hipSuccess;
  }

  hipError_t recordEvent(void* ptr, at::hip::HIPStreamMasqueradingAsCUDA stream)
  {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = blocks.find(ptr);
    if (it == blocks.end()) {
      // ignore events for untracked pointers
      return hipSuccess;
    }

    Block& block = it->second;
    THAssert(block.allocated);

    block.streams.insert(stream);
    return hipSuccess;
  }

  hipError_t processEvents()
  {
    // Process outstanding cudaEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!cuda_events.empty()) {
      auto& e = cuda_events.front();
      hipEvent_t event = e.first;

      hipError_t err = hipEventQuery(event);
      if (err == hipErrorNotReady) {
        // ignore and clear the error if not ready
        hipGetLastError();
        break;
      } else if (err != hipSuccess) {
        return err;
      }
      err = hipEventDestroy(event);
      if (err != hipSuccess) {
        return err;
      }

      Block& block = blocks.at(e.second);
      block.event_count--;
      if (block.event_count == 0 && !block.allocated) {
        available.insert(block);
      }
      cuda_events.pop_front();
    }
    return hipSuccess;
  }

  void emptyCache()
  {
    std::lock_guard<std::mutex> lock(mutex);

    // remove events for freed blocks
    for (const auto & cuda_event : cuda_events) {
      const hipEvent_t event = cuda_event.first;
      Block& block = blocks.at(cuda_event.second);
      if (!block.allocated) {
        THCudaCheckWarn(hipEventDestroy(event));
        block.event_count--;
      }
    }

    // all cuda_events have been processed
    cuda_events.clear();

    // clear list of available blocks
    available.clear();

    // free and erase non-allocated blocks
    for (auto it = blocks.begin(); it != blocks.end();) {
      Block& block = it->second;
      if (!block.allocated) {
        THCudaCheckWarn(hipHostFree(block.ptr));
        it = blocks.erase(it);
      } else {
        ++it;
      }
    }
  }

  hipError_t insertEvents(Block& block)
  {
    hipError_t err;

    int prev_device;
    err = hipGetDevice(&prev_device);
    if (err != hipSuccess) return err;

    std::unordered_set<at::hip::HIPStreamMasqueradingAsCUDA> streams(std::move(block.streams));
    for (auto stream : streams) {
      err = hipSetDevice(stream.device_index());
      if (err != hipSuccess) break;

      hipEvent_t event;
      err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
      if (err != hipSuccess) break;

      err = hipEventRecord(event, stream.stream());
      if (err != hipSuccess) break;

      block.event_count++;
      cuda_events.emplace_back(event, block.ptr);
    }

    hipSetDevice(prev_device);
    return err;
  }
};

}  // namespace

static HostAllocator allocator;

hipError_t THCCachingHostAllocator_recordEvent(void *ptr, at::hip::HIPStreamMasqueradingAsCUDA stream)
{
  return allocator.recordEvent(ptr, stream);
}

void THCCachingHostAllocator_emptyCache()
{
  allocator.emptyCache();
}

static void THCCachingHostDeleter(void* ptr) {
  allocator.free(ptr);
}

struct THCCachingHostAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void *ptr;
    THCudaCheck(allocator.malloc(&ptr, size));
    return {ptr, ptr, &THCCachingHostDeleter, at::DeviceType::CPU};
  }
  at::DeleterFnPtr raw_deleter() const override {
    return &THCCachingHostDeleter;
  }
};

static THCCachingHostAllocator thc_caching_host_allocator;
at::Allocator* getTHCCachingHostAllocator() {
  return &thc_caching_host_allocator;
}
