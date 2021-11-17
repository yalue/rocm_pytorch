#include <aten/src/ATen/cuda/CUDAEvent.h>
#include <c10/core/Device.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/custom_class.h>

namespace torch {
namespace jit {

class CUDAEvent;
// This class is a wrapper around c10::hip::HIPStreamMasqueradingAsCUDA.
// It is needed because TorchBind does not support all of the argument types
// for c10::hip::HIPStreamMasqueradingAsCUDA. For more details, please refer to
// ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h.
class HIPStreamMasqueradingAsCUDA final : public CustomClassHolder {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  HIPStreamMasqueradingAsCUDA(
      c10::optional<c10::Device> device = c10::nullopt,
      int64_t priority = 0) {
    constexpr int64_t PRIORITY_INDEX = 0;
    c10::DeviceIndex device_index =
        device.has_value() ? device->index() : c10::hip::current_device();
    stream_ = std::make_unique<c10::hip::HIPStreamMasqueradingAsCUDA>(
        c10::hip::getStreamFromPoolMasqueradingAsCUDA(priority < PRIORITY_INDEX, device_index));
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  HIPStreamMasqueradingAsCUDA(c10::hip::HIPStreamMasqueradingAsCUDA s) {
    stream_ = std::make_unique<c10::hip::HIPStreamMasqueradingAsCUDA>(s);
  }

  bool query() {
    return stream_->query();
  }

  c10::intrusive_ptr<CUDAEvent> recordEvent(
      c10::intrusive_ptr<CUDAEvent> event);

  void synchronize() {
    stream_->synchronize();
  }

  void waitEvent(c10::intrusive_ptr<CUDAEvent> event);

  void waitStream(c10::intrusive_ptr<HIPStreamMasqueradingAsCUDA> stream);

  /// Get the CUDA device index that this stream is associated with.
  int64_t device_index() const {
    return stream_->device_index();
  }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a CUDA device.
  c10::Device device() const {
    return stream_->device();
  }

  /// Return the stream ID corresponding to this particular stream.
  int64_t id() const {
    return stream_->id();
  }

  /// Pack a HIPStreamMasqueradingAsCUDA to uint64_t representation.
  /// The HIPStreamMasqueradingAsCUDA can be unpacked using unpack().  The format of
  /// the uint64_t is unspecified and may be changed.
  int64_t pack() const {
    return stream_->pack();
  }

 private:
  std::unique_ptr<c10::hip::HIPStreamMasqueradingAsCUDA> stream_;
  friend class CUDAEvent;
};

// This class is a wrapper around at::hip::HIPStreamMasqueradingAsCUDA.
// It is needed because TorchBind does not support all of the argument types
// for at::cuda::CUDAEvent. For more details, please refer to
// aten/src/ATen/cuda/CUDAEvent.h.
class CUDAEvent final : public CustomClassHolder {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CUDAEvent(
      bool enable_timing = false,
      bool blocking = false,
      bool interprocess = false) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int flags = hipEventDisableTiming;
    if (enable_timing) {
      flags = hipEventDefault;
    }
    if (blocking) {
      flags |= hipEventBlockingSync;
    }
    if (interprocess) {
      TORCH_CHECK(!enable_timing);
      flags |= hipEventInterprocess;
    }

    event_ = std::make_unique<at::cuda::CUDAEvent>(flags);
  }

  double elapsedTime(c10::intrusive_ptr<CUDAEvent> end) {
    return event_->elapsed_time(*end->event_);
  }

  std::string ipcHandle() {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    hipIpcEventHandle_t handle;
    event_->ipc_handle(&handle);
    std::string str_handle((const char*)&handle, sizeof(handle));
    return str_handle;
  }

  bool query() {
    return event_->query();
  }

  void record(c10::intrusive_ptr<HIPStreamMasqueradingAsCUDA> stream);

  void synchronize() {
    event_->synchronize();
  }
  void wait(c10::intrusive_ptr<HIPStreamMasqueradingAsCUDA> stream);

 private:
  void recordInternal(HIPStreamMasqueradingAsCUDA* stream);
  std::unique_ptr<at::cuda::CUDAEvent> event_;

  friend class HIPStreamMasqueradingAsCUDA;
};

c10::intrusive_ptr<CUDAEvent> HIPStreamMasqueradingAsCUDA::recordEvent(
    c10::intrusive_ptr<CUDAEvent> event) {
  if (!event) {
    event = c10::make_intrusive<CUDAEvent>();
  }

  event->recordInternal(this);
  return event;
}

void HIPStreamMasqueradingAsCUDA::waitEvent(c10::intrusive_ptr<CUDAEvent> event) {
  event->event_->block(*stream_);
}

void HIPStreamMasqueradingAsCUDA::waitStream(c10::intrusive_ptr<HIPStreamMasqueradingAsCUDA> stream) {
  auto ev = c10::make_intrusive<CUDAEvent>();
  stream->recordEvent(ev);
  waitEvent(ev);
}

void CUDAEvent::record(c10::intrusive_ptr<HIPStreamMasqueradingAsCUDA> stream) {
  event_->record(*stream->stream_);
}

void CUDAEvent::recordInternal(HIPStreamMasqueradingAsCUDA* stream) {
  event_->record(*stream->stream_);
}

void CUDAEvent::wait(c10::intrusive_ptr<HIPStreamMasqueradingAsCUDA> stream) {
  event_->block(*stream->stream_);
}

TORCH_LIBRARY(cuda, m) {
  auto stream_class = m.class_<torch::jit::HIPStreamMasqueradingAsCUDA>("Stream").def(
      torch::init<c10::optional<c10::Device>, int64_t>(),
      "",
      {torch::arg("device") = c10::nullopt, torch::arg("priority") = 0});
  auto event_class = m.class_<torch::jit::CUDAEvent>("Event").def(
      torch::init<bool, bool, bool>(),
      "",
      {torch::arg("enable_timing") = false,
       torch::arg("blocking") = false,
       torch::arg("interprocess") = false});

  stream_class.def("query", &HIPStreamMasqueradingAsCUDA::query)
      .def("record_event", &HIPStreamMasqueradingAsCUDA::recordEvent)
      .def("synchronize", &HIPStreamMasqueradingAsCUDA::synchronize)
      .def("wait_event", &HIPStreamMasqueradingAsCUDA::waitEvent)
      .def("wait_stream", &HIPStreamMasqueradingAsCUDA::waitStream)
      .def("device_index", &HIPStreamMasqueradingAsCUDA::device_index)
      .def_property("device", &HIPStreamMasqueradingAsCUDA::device)
      .def("pack", &HIPStreamMasqueradingAsCUDA::pack)
      .def("id", &HIPStreamMasqueradingAsCUDA::id);

  event_class.def("elapsed_time", &CUDAEvent::elapsedTime)
      .def("query", &CUDAEvent::query)
      .def("record", &CUDAEvent::record)
      .def("synchronize", &CUDAEvent::synchronize)
      .def("wait", &CUDAEvent::wait);
};

} // namespace jit
} // namespace torch
