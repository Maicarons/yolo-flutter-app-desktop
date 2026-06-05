// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef FLUTTER_PLUGIN_ULTRALYTICS_YOLO_CAMERA_CAPTURE_LINUX_H_
#define FLUTTER_PLUGIN_ULTRALYTICS_YOLO_CAMERA_CAPTURE_LINUX_H_

#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace ultralytics_yolo {

struct CameraDeviceInfo {
  std::string id;
  std::string name;
};

using FrameCallback =
    std::function<void(const uint8_t* data, int width, int height)>;

/// Captures camera frames using Video4Linux2 (V4L2).
class CameraCapture {
 public:
  CameraCapture();
  ~CameraCapture();

  static std::vector<CameraDeviceInfo> EnumerateDevices();

  void Start(int deviceIndex, const std::string& resolution,
             FrameCallback callback);

  void Stop();
  bool IsCapturing() const { return capturing_; }

 private:
  void ReadLoop();

  bool capturing_ = false;
  bool stopRequested_ = false;
  FrameCallback callback_;
  int fd_ = -1;
  std::thread readThread_;
};

}  // namespace ultralytics_yolo

#endif
