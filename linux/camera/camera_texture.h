// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef FLUTTER_PLUGIN_ULTRALYTICS_YOLO_CAMERA_TEXTURE_LINUX_H_
#define FLUTTER_PLUGIN_ULTRALYTICS_YOLO_CAMERA_TEXTURE_LINUX_H_

#include <flutter/plugin_registrar.h>
#include <flutter/texture_registrar.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace ultralytics_yolo {

class CameraTexture {
 public:
  CameraTexture(flutter::TextureRegistrar* textureRegistrar);
  ~CameraTexture();

  int64_t Register();
  void Unregister();
  void UpdateFrame(const uint8_t* rgbData, int width, int height);
  int64_t textureId() const { return textureId_; }

 private:
  flutter::TextureRegistrar* registrar_;
  std::unique_ptr<flutter::TextureVariant> texture_;
  int64_t textureId_ = -1;

  std::vector<uint8_t> frameBuffer_;
  int frameWidth_ = 0;
  int frameHeight_ = 0;
  std::mutex mutex_;
};

}  // namespace ultralytics_yolo

#endif
