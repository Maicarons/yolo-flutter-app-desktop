// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef FLUTTER_PLUGIN_ULTRALYTICS_YOLO_PLUGIN_LINUX_H_
#define FLUTTER_PLUGIN_ULTRALYTICS_YOLO_PLUGIN_LINUX_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar.h>
#include <flutter/standard_method_codec.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "camera/camera_capture.h"
#include "camera/camera_texture.h"
#include "inference/yolo_engine.h"

namespace ultralytics_yolo {

class YoloPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarLinux* registrar);

  YoloPlugin(flutter::PluginRegistrarLinux* registrar,
             std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>>
                 channel);

  virtual ~YoloPlugin();

 private:
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleLoadModel(
      const flutter::EncodableMap& args,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandlePredictSingleImage(
      const flutter::EncodableMap& args,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleInspectModel(
      const flutter::EncodableMap& args,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleCheckModelExists(
      const flutter::EncodableMap& args,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleGetStoragePaths(
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleCreateInstance(
      const flutter::EncodableMap& args,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleDisposeInstance(
      const flutter::EncodableMap& args,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandlePredictorInstance(
      const flutter::EncodableMap& args,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleSetModel(
      const flutter::EncodableMap& args,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  void HandleGetPlatformVersion(
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  std::string ResolveModelPath(const std::string& modelPath);
  std::string GetAppSupportDir();

  flutter::PluginRegistrarLinux* registrar_;
  std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> channel_;
  std::unordered_map<std::string, std::unique_ptr<YoloEngine>> engines_;
  std::unique_ptr<CameraCapture> camera_;
  std::unique_ptr<CameraTexture> camera_texture_;
};

}  // namespace ultralytics_yolo

#endif
