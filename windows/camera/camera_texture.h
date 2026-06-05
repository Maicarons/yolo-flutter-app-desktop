// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef FLUTTER_PLUGIN_ULTRALYTICS_YOLO_CAMERA_TEXTURE_H_
#define FLUTTER_PLUGIN_ULTRALYTICS_YOLO_CAMERA_TEXTURE_H_

#include <flutter/plugin_registrar_windows.h>
#include <flutter/texture_registrar.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace ultralytics_yolo {

/// Manages a Flutter texture that displays camera frames with YOLO overlays.
class CameraTexture {
 public:
  CameraTexture(flutter::TextureRegistrar* textureRegistrar);
  ~CameraTexture();

  /// Register the texture with Flutter.  Returns the texture ID.
  int64_t Register();

  /// Unregister the texture.
  void Unregister();

  /// Update the texture with a new BGRA frame.
  void UpdateFrame(const uint8_t* bgraData, int width, int height);

  int64_t textureId() const { return textureId_; }

 private:
  flutter::TextureRegistrar* registrar_;
  std::unique_ptr<flutter::TextureVariant> texture_;
  int64_t textureId_ = -1;

  // Current frame buffer (BGRA).
  std::vector<uint8_t> frameBuffer_;
  int frameWidth_ = 0;
  int frameHeight_ = 0;
  std::mutex mutex_;
};

}  // namespace ultralytics_yolo

#endif  // FLUTTER_PLUGIN_ULTRALYTICS_YOLO_CAMERA_TEXTURE_H_
