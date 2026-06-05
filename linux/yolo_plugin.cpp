// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "yolo_plugin.h"

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar.h>
#include <flutter/standard_method_codec.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <sys/utsname.h>

#include "inference/model_metadata.h"

namespace ultralytics_yolo {

// ── Registration ─────────────────────────────────────────────────────────────

void YoloPlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarLinux* registrar) {
  auto channel =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          registrar->messenger(), "yolo_single_image_channel",
          &flutter::StandardMethodCodec::GetInstance());

  auto* channel_ptr = channel.get();
  auto plugin = std::make_unique<YoloPlugin>(registrar, std::move(channel));

  channel_ptr->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto& call, auto result) {
        plugin_pointer->HandleMethodCall(call, std::move(result));
      });

  registrar->AddPlugin(std::move(plugin));
}

// ── Constructor / Destructor ─────────────────────────────────────────────────

YoloPlugin::YoloPlugin(
    flutter::PluginRegistrarLinux* registrar,
    std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> channel)
    : registrar_(registrar), channel_(std::move(channel)) {}

YoloPlugin::~YoloPlugin() = default;

// ── Helpers ──────────────────────────────────────────────────────────────────

std::string YoloPlugin::GetAppSupportDir() {
  // Use XDG_DATA_HOME or fallback to ~/.local/share
  const char* xdg = std::getenv("XDG_DATA_HOME");
  std::string base;
  if (xdg && xdg[0]) {
    base = xdg;
  } else {
    const char* home = std::getenv("HOME");
    base = home ? std::string(home) + "/.local/share" : "/tmp";
  }
  std::string result = base + "/ultralytics_yolo";
  std::filesystem::create_directories(result);
  return result;
}

std::string YoloPlugin::ResolveModelPath(const std::string& modelPath) {
  if (!modelPath.empty() && modelPath[0] == '/') return modelPath;
  if (modelPath.rfind("internal://", 0) == 0)
    return GetAppSupportDir() + "/" + modelPath.substr(11);
  return GetAppSupportDir() + "/" + modelPath;
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

void YoloPlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue>& call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  const auto* args = std::get_if<flutter::EncodableMap>(call.arguments());
  auto args_or_empty = args ? *args : flutter::EncodableMap{};

  if (call.method_name() == "loadModel") {
    HandleLoadModel(args_or_empty, std::move(result));
  } else if (call.method_name() == "predictSingleImage") {
    HandlePredictSingleImage(args_or_empty, std::move(result));
  } else if (call.method_name() == "inspectModel") {
    HandleInspectModel(args_or_empty, std::move(result));
  } else if (call.method_name() == "checkModelExists") {
    HandleCheckModelExists(args_or_empty, std::move(result));
  } else if (call.method_name() == "getStoragePaths") {
    HandleGetStoragePaths(std::move(result));
  } else if (call.method_name() == "createInstance") {
    HandleCreateInstance(args_or_empty, std::move(result));
  } else if (call.method_name() == "disposeInstance") {
    HandleDisposeInstance(args_or_empty, std::move(result));
  } else if (call.method_name() == "predictorInstance") {
    HandlePredictorInstance(args_or_empty, std::move(result));
  } else if (call.method_name() == "setModel") {
    HandleSetModel(args_or_empty, std::move(result));
  } else if (call.method_name() == "getPlatformVersion") {
    HandleGetPlatformVersion(std::move(result));
  } else {
    result->NotImplemented();
  }
}

void YoloPlugin::HandleGetPlatformVersion(
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  struct utsname unameData {};
  uname(&unameData);
  std::ostringstream ss;
  ss << "Linux " << unameData.release << " (" << unameData.machine << ")";
  result->Success(flutter::EncodableValue(ss.str()));
}

void YoloPlugin::HandleLoadModel(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  std::string modelPath = "yolo26n";
  std::string taskStr = "detect";
  std::string instanceId = "default";
  bool useGpu = true;

  auto it = args.find(flutter::EncodableValue("modelPath"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) modelPath = *v;
  }
  it = args.find(flutter::EncodableValue("task"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) taskStr = *v;
  }
  it = args.find(flutter::EncodableValue("instanceId"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) instanceId = *v;
  }
  it = args.find(flutter::EncodableValue("useGpu"));
  if (it != args.end()) {
    if (auto* v = std::get_if<bool>(&it->second)) useGpu = *v;
  }

  std::string resolvedPath = ResolveModelPath(modelPath);
  auto& engine = engines_[instanceId];
  if (!engine) engine = std::make_unique<YoloEngine>();

  try {
    engine->LoadModel(resolvedPath, taskStr, useGpu);
    result->Success(flutter::EncodableValue(true));
  } catch (const std::exception& e) {
    result->Error("MODEL_NOT_FOUND", e.what());
  }
}

void YoloPlugin::HandlePredictSingleImage(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  std::vector<uint8_t> imageData;
  std::string instanceId = "default";
  double confidenceThreshold = -1.0;
  double iouThreshold = -1.0;

  auto it = args.find(flutter::EncodableValue("image"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::vector<uint8_t>>(&it->second))
      imageData = *v;
  }
  it = args.find(flutter::EncodableValue("instanceId"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) instanceId = *v;
  }
  it = args.find(flutter::EncodableValue("confidenceThreshold"));
  if (it != args.end()) {
    if (auto* v = std::get_if<double>(&it->second)) confidenceThreshold = *v;
  }
  it = args.find(flutter::EncodableValue("iouThreshold"));
  if (it != args.end()) {
    if (auto* v = std::get_if<double>(&it->second)) iouThreshold = *v;
  }

  if (imageData.empty()) {
    result->Error("bad_args", "No image data");
    return;
  }

  auto eit = engines_.find(instanceId);
  if (eit == engines_.end() || !eit->second) {
    result->Error("MODEL_NOT_LOADED",
                  "Model has not been loaded. Call loadModel() first.");
    return;
  }

  try {
    auto response =
        eit->second->Predict(imageData, confidenceThreshold, iouThreshold);
    result->Success(flutter::EncodableValue(response));
  } catch (const std::exception& e) {
    result->Error("prediction_error", e.what());
  }
}

void YoloPlugin::HandleInspectModel(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  std::string modelPath;
  auto it = args.find(flutter::EncodableValue("modelPath"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) modelPath = *v;
  }
  std::string resolvedPath = ResolveModelPath(modelPath);

  try {
    auto metadata = ModelMetadata::Inspect(resolvedPath);
    result->Success(flutter::EncodableValue(metadata));
  } catch (...) {
    flutter::EncodableMap fallback;
    fallback[flutter::EncodableValue("path")] = resolvedPath;
    fallback[flutter::EncodableValue("task")] = std::string("");
    fallback[flutter::EncodableValue("labels")] = flutter::EncodableList{};
    result->Success(flutter::EncodableValue(fallback));
  }
}

void YoloPlugin::HandleCheckModelExists(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  std::string modelPath;
  auto it = args.find(flutter::EncodableValue("modelPath"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) modelPath = *v;
  }
  std::string resolvedPath = ResolveModelPath(modelPath);
  bool exists = std::filesystem::exists(resolvedPath);

  flutter::EncodableMap response;
  response[flutter::EncodableValue("exists")] = exists;
  response[flutter::EncodableValue("path")] = resolvedPath;
  response[flutter::EncodableValue("location")] =
      exists ? std::string("local") : std::string("not_found");
  result->Success(flutter::EncodableValue(response));
}

void YoloPlugin::HandleGetStoragePaths(
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  std::string appSupport = GetAppSupportDir();
  std::string cacheDir = appSupport + "/cache";
  std::filesystem::create_directories(cacheDir);

  flutter::EncodableMap paths;
  paths[flutter::EncodableValue("internal")] = appSupport;
  paths[flutter::EncodableValue("cache")] = cacheDir;
  paths[flutter::EncodableValue("external")] = std::string();
  paths[flutter::EncodableValue("externalCache")] = std::string();
  result->Success(flutter::EncodableValue(paths));
}

void YoloPlugin::HandleCreateInstance(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  std::string instanceId;
  auto it = args.find(flutter::EncodableValue("instanceId"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) instanceId = *v;
  }
  if (instanceId.empty()) {
    result->Error("bad_args", "Missing instanceId");
    return;
  }
  engines_[instanceId] = nullptr;
  result->Success(nullptr);
}

void YoloPlugin::HandleDisposeInstance(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  std::string instanceId;
  auto it = args.find(flutter::EncodableValue("instanceId"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) instanceId = *v;
  }
  engines_.erase(instanceId);
  result->Success(nullptr);
}

void YoloPlugin::HandlePredictorInstance(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  std::string instanceId = "default";
  auto it = args.find(flutter::EncodableValue("instanceId"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) instanceId = *v;
  }
  auto eit = engines_.find(instanceId);
  if (eit == engines_.end() || !eit->second) {
    result->Error("predictor_instance_error", "Engine not initialized");
    return;
  }
  try {
    eit->second->WarmUp();
    result->Success(nullptr);
  } catch (const std::exception& e) {
    result->Error("predictor_instance_error", e.what());
  }
}

void YoloPlugin::HandleSetModel(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  std::string modelPath;
  std::string taskStr;
  bool useGpu = true;

  auto it = args.find(flutter::EncodableValue("modelPath"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) modelPath = *v;
  }
  it = args.find(flutter::EncodableValue("task"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) taskStr = *v;
  }
  it = args.find(flutter::EncodableValue("useGpu"));
  if (it != args.end()) {
    if (auto* v = std::get_if<bool>(&it->second)) useGpu = *v;
  }

  std::string resolvedPath = ResolveModelPath(modelPath);
  auto& engine = engines_["default"];
  if (!engine) engine = std::make_unique<YoloEngine>();

  try {
    engine->LoadModel(resolvedPath, taskStr, useGpu);
    result->Success(nullptr);
  } catch (const std::exception& e) {
    result->Error("MODEL_NOT_FOUND", e.what());
  }
}

}  // namespace ultralytics_yolo

// ── C entry point ────────────────────────────────────────────────────────────

void YoloPluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  ultralytics_yolo::YoloPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarLinux>(registrar));
}
