// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef FLUTTER_PLUGIN_ULTRALYTICS_YOLO_CAMERA_CAPTURE_H_
#define FLUTTER_PLUGIN_ULTRALYTICS_YOLO_CAMERA_CAPTURE_H_

#include <d3d11.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ultralytics_yolo {

/// Callback signature: receives raw BGRA frame data + dimensions.
using FrameCallback =
    std::function<void(const uint8_t* data, int width, int height)>;

/// Lists available camera devices.
struct CameraDeviceInfo {
  std::wstring id;
  std::wstring name;
};

/// Captures camera frames using Windows MediaFoundation.
class CameraCapture {
 public:
  CameraCapture();
  ~CameraCapture();

  /// Enumerate available camera devices.
  static std::vector<CameraDeviceInfo> EnumerateDevices();

  /// Start capturing from the camera with the given device index.
  /// `resolution` is "720p", "1080p", etc.
  void Start(int deviceIndex, const std::string& resolution,
             FrameCallback callback);

  /// Stop capturing.
  void Stop();

  bool IsCapturing() const { return capturing_; }

 private:
  void Cleanup();

  bool capturing_ = false;
  FrameCallback callback_;

  Microsoft::WRL::ComPtr<IMFSourceReader> reader_;
  Microsoft::WRL::ComPtr<IMFMediaSource> source_;
  Microsoft::WRL::ComPtr<IMFAttributes> attrs_;

  std::mutex mutex_;
  HANDLE readThread_ = nullptr;
  bool stopRequested_ = false;

  // Thread proc.
  static DWORD WINAPI ReadThreadProc(LPVOID param);
  void ReadLoop();
};

}  // namespace ultralytics_yolo

#endif  // FLUTTER_PLUGIN_ULTRALYTICS_YOLO_CAMERA_CAPTURE_H_
