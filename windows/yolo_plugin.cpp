// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "yolo_plugin.h"

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>
#include <windows.h>
#include <shlobj.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "inference/model_metadata.h"

namespace ultralytics_yolo {

// ── Registration ─────────────────────────────────────────────────────────────

void YoloPlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows* registrar) {
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
    flutter::PluginRegistrarWindows* registrar,
    std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> channel)
    : registrar_(registrar), channel_(std::move(channel)) {}

YoloPlugin::~YoloPlugin() = default;

// ── Helpers ──────────────────────────────────────────────────────────────────

std::string YoloPlugin::GetAppSupportDir() {
  wchar_t* path = nullptr;
  if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0, nullptr,
                                      &path))) {
    std::wstring ws(path);
    CoTaskMemFree(path);
    // Convert wide string to narrow string.
    int size = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, nullptr, 0,
                                   nullptr, nullptr);
    std::string result(size - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, result.data(), size,
                        nullptr, nullptr);
    result += "\\ultralytics_yolo";
    std::filesystem::create_directories(result);
    return result;
  }
  return ".";
}

std::string YoloPlugin::ResolveModelPath(const std::string& modelPath) {
  // Absolute path — use as-is.
  if (!modelPath.empty() && (modelPath[1] == ':' || modelPath[0] == '\\')) {
    return modelPath;
  }
  // internal:// scheme — resolve relative to app support dir.
  if (modelPath.rfind("internal://", 0) == 0) {
    return GetAppSupportDir() + "\\" + modelPath.substr(11);
  }
  // Otherwise treat as a relative path in app support dir.
  return GetAppSupportDir() + "\\" + modelPath;
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
  } else if (call.method_name() == "startDesktopCamera") {
    HandleStartCamera(args_or_empty, std::move(result));
  } else if (call.method_name() == "stopDesktopCamera") {
    HandleStopCamera(args_or_empty, std::move(result));
  } else if (call.method_name() == "getPlatformVersion") {
    HandleGetPlatformVersion(std::move(result));
  } else {
    result->NotImplemented();
  }
}

// ── getPlatformVersion ───────────────────────────────────────────────────────

void YoloPlugin::HandleGetPlatformVersion(
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  OSVERSIONINFOW osvi = {};
  osvi.dwOSVersionInfoSize = sizeof(osvi);
  // RtlGetVersion is always available and never lies about compatibility mode.
  auto rtl = reinterpret_cast<NTSTATUS(WINAPI*)(PRTL_OSVERSIONINFOW)>(
      GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion"));
  if (rtl) rtl(&osvi);
  std::ostringstream ss;
  ss << "Windows " << osvi.dwMajorVersion << "." << osvi.dwMinorVersion
     << " (build " << osvi.dwBuildNumber << ")";
  result->Success(flutter::EncodableValue(ss.str()));
}

// ── loadModel ────────────────────────────────────────────────────────────────

void YoloPlugin::HandleLoadModel(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  // Extract arguments.
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

  // Get or create engine for this instance.
  auto& engine = engines_[instanceId];
  if (!engine) {
    engine = std::make_unique<YoloEngine>();
  }

  try {
    engine->LoadModel(resolvedPath, taskStr, useGpu);
    result->Success(flutter::EncodableValue(true));
  } catch (const std::exception& e) {
    result->Error("MODEL_NOT_FOUND", e.what());
  }
}

// ── predictSingleImage ───────────────────────────────────────────────────────

void YoloPlugin::HandlePredictSingleImage(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  // Extract image bytes.
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
    auto response = eit->second->Predict(imageData, confidenceThreshold,
                                         iouThreshold);
    result->Success(flutter::EncodableValue(response));
  } catch (const std::exception& e) {
    result->Error("prediction_error", e.what());
  }
}

// ── inspectModel ─────────────────────────────────────────────────────────────

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
    // Return a default empty metadata map on failure (matches Android).
    flutter::EncodableMap fallback;
    fallback[flutter::EncodableValue("path")] = resolvedPath;
    fallback[flutter::EncodableValue("task")] = std::string("");
    fallback[flutter::EncodableValue("labels")] =
        flutter::EncodableList{};
    result->Success(flutter::EncodableValue(fallback));
  }
}

// ── checkModelExists ─────────────────────────────────────────────────────────

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

// ── getStoragePaths ──────────────────────────────────────────────────────────

void YoloPlugin::HandleGetStoragePaths(
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  std::string appSupport = GetAppSupportDir();

  // Cache is a subdirectory.
  std::string cacheDir = appSupport + "\\cache";
  std::filesystem::create_directories(cacheDir);

  flutter::EncodableMap paths;
  paths[flutter::EncodableValue("internal")] = appSupport;
  paths[flutter::EncodableValue("cache")] = cacheDir;
  paths[flutter::EncodableValue("external")] = std::string();
  paths[flutter::EncodableValue("externalCache")] = std::string();
  result->Success(flutter::EncodableValue(paths));
}

// ── createInstance ───────────────────────────────────────────────────────────

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

  // Pre-create an empty slot so later loadModel finds it.
  engines_[instanceId] = nullptr;
  result->Success(nullptr);
}

// ── disposeInstance ──────────────────────────────────────────────────────────

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

// ── predictorInstance ────────────────────────────────────────────────────────

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

  // Warm-up: run a dummy inference to pre-allocate tensors.
  try {
    eit->second->WarmUp();
    result->Success(nullptr);
  } catch (const std::exception& e) {
    result->Error("predictor_instance_error", e.what());
  }
}

// ── setModel (platform view model switch) ────────────────────────────────────

void YoloPlugin::HandleSetModel(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  // For now, delegate to the "default" engine.
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

// ── startDesktopCamera ───────────────────────────────────────────────────────

void YoloPlugin::HandleStartCamera(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  std::string viewId;
  std::string modelPath = "yolo26n";
  std::string taskStr = "detect";

  auto it = args.find(flutter::EncodableValue("viewId"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) viewId = *v;
  }
  it = args.find(flutter::EncodableValue("modelPath"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) modelPath = *v;
  }
  it = args.find(flutter::EncodableValue("task"));
  if (it != args.end()) {
    if (auto* v = std::get_if<std::string>(&it->second)) taskStr = *v;
  }

  try {
    // Register texture with Flutter.
    auto* texture_registrar = registrar_->texture_registrar();
    camera_texture_ = std::make_unique<CameraTexture>(texture_registrar);
    int64_t texture_id = camera_texture_->Register();

    // Create and load engine for this camera view.
    auto& engine = engines_[viewId.empty() ? "default" : viewId];
    if (!engine) engine = std::make_unique<YoloEngine>();

    // Load model if path is provided.
    std::string resolvedPath = ResolveModelPath(modelPath);
    engine->LoadModel(resolvedPath, taskStr, true);

    // Start camera capture.
    camera_ = std::make_unique<CameraCapture>();
    camera_->Start(0, "720p", [this](const uint8_t* data, int w, int h) {
      // Push frame to Flutter texture.
      if (camera_texture_) {
        camera_texture_->UpdateFrame(data, w, h);
      }
    });

    // Return texture ID to Dart.
    flutter::EncodableMap response;
    response[flutter::EncodableValue("textureId")] = texture_id;
    result->Success(flutter::EncodableValue(response));
  } catch (const std::exception& e) {
    result->Error("CAMERA_ERROR", e.what());
  }
}

// ── stopDesktopCamera ────────────────────────────────────────────────────────

void YoloPlugin::HandleStopCamera(
    const flutter::EncodableMap& args,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  if (camera_) {
    camera_->Stop();
    camera_.reset();
  }
  if (camera_texture_) {
    camera_texture_->Unregister();
    camera_texture_.reset();
  }
  result->Success(nullptr);
}

}  // namespace ultralytics_yolo

// ── C entry point ────────────────────────────────────────────────────────────

void YoloPluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  ultralytics_yolo::YoloPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
