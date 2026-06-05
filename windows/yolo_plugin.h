// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef FLUTTER_PLUGIN_ULTRALYTICS_YOLO_PLUGIN_H_
#define FLUTTER_PLUGIN_ULTRALYTICS_YOLO_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "inference/yolo_engine.h"
#include "camera/camera_capture.h"
#include "camera/camera_texture.h"

namespace ultralytics_yolo {

class YoloPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar);

  YoloPlugin(
      flutter::PluginRegistrarWindows* registrar,
      std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> channel);

  virtual ~YoloPlugin();

 private:
  // ── MethodChannel handlers ──────────────────────────────────────────────
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  // ── Individual method implementations ───────────────────────────────────
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

  // ── Helpers ─────────────────────────────────────────────────────────────
  std::string ResolveModelPath(const std::string& modelPath);
  std::string GetAppSupportDir();

  // ── Members ─────────────────────────────────────────────────────────────
  flutter::PluginRegistrarWindows* registrar_;
  std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> channel_;

  // Per-instance engines keyed by instanceId ("default" for single-instance).
  std::unordered_map<std::string, std::unique_ptr<YoloEngine>> engines_;

  // Camera subsystem (one active camera at a time).
  std::unique_ptr<CameraCapture> camera_;
  std::unique_ptr<CameraTexture> camera_texture_;
};

}  // namespace ultralytics_yolo

// C entry point for Flutter plugin registration.
extern "C" __declspec(dllexport) void YoloPluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar);

#endif  // FLUTTER_PLUGIN_ULTRALYTICS_YOLO_PLUGIN_H_
