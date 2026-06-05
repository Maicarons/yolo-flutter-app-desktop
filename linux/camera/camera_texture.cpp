// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "camera_texture.h"

#include <cstring>

namespace ultralytics_yolo {

CameraTexture::CameraTexture(flutter::TextureRegistrar* textureRegistrar)
    : registrar_(textureRegistrar) {}

CameraTexture::~CameraTexture() { Unregister(); }

int64_t CameraTexture::Register() {
  if (textureId_ >= 0) return textureId_;

  texture_ = std::make_unique<flutter::TextureVariant>(
      flutter::PixelBufferTexture([this](size_t width, size_t height)
                                      -> const FlutterDesktopPixelBuffer* {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frameBuffer_.empty()) return nullptr;

        static thread_local FlutterDesktopPixelBuffer pixelBuffer{};
        pixelBuffer.buffer = frameBuffer_.data();
        pixelBuffer.width = static_cast<size_t>(frameWidth_);
        pixelBuffer.height = static_cast<size_t>(frameHeight_);
        pixelBuffer.release_context = nullptr;
        pixelBuffer.release_callback = nullptr;
        return &pixelBuffer;
      }));

  textureId_ = registrar_->RegisterTexture(texture_.get());
  return textureId_;
}

void CameraTexture::Unregister() {
  if (textureId_ >= 0) {
    registrar_->UnregisterTexture(textureId_);
    textureId_ = -1;
  }
}

void CameraTexture::UpdateFrame(const uint8_t* rgbData, int width, int height) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t dataSize = static_cast<size_t>(width) * height * 4;
    if (frameWidth_ != width || frameHeight_ != height) {
      frameBuffer_.resize(dataSize);
      frameWidth_ = width;
      frameHeight_ = height;
    }
    std::memcpy(frameBuffer_.data(), rgbData, dataSize);
  }

  if (textureId_ >= 0) {
    registrar_->MarkTextureFrameAvailable(textureId_);
  }
}

}  // namespace ultralytics_yolo
