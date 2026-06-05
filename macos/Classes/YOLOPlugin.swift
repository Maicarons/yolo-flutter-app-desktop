// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Cocoa
import FlutterMacOS
import Foundation

public class YOLOPlugin: NSObject, FlutterPlugin {
  private var engines: [String: YoloEngine] = [:]

  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(
      name: "yolo_single_image_channel",
      binaryMessenger: registrar.messenger
    )
    let instance = YOLOPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any] else {
      switch call.method {
      case "getStoragePaths":
        handleGetStoragePaths(result: result)
      default:
        result(FlutterError(code: "bad_args", message: "Arguments required", details: nil))
      }
      return
    }

    switch call.method {
    case "loadModel":
      handleLoadModel(args: args, result: result)
    case "predictSingleImage":
      handlePredictSingleImage(args: args, result: result)
    case "inspectModel":
      handleInspectModel(args: args, result: result)
    case "checkModelExists":
      handleCheckModelExists(args: args, result: result)
    case "getStoragePaths":
      handleGetStoragePaths(result: result)
    case "createInstance":
      handleCreateInstance(args: args, result: result)
    case "disposeInstance":
      handleDisposeInstance(args: args, result: result)
    case "predictorInstance":
      handlePredictorInstance(args: args, result: result)
    case "setModel":
      handleSetModel(args: args, result: result)
    case "getPlatformVersion":
      result("macOS \(ProcessInfo.processInfo.operatingSystemVersionString)")
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  // MARK: - Helpers

  private func getAppSupportDir() -> String {
    let paths = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
    let appSupport = paths.first?.path ?? NSTemporaryDirectory()
    let dir = (appSupport as NSString).appendingPathComponent("ultralytics_yolo")
    try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    return dir
  }

  private func resolveModelPath(_ modelPath: String) -> String {
    if modelPath.hasPrefix("/") { return modelPath }
    if modelPath.hasPrefix("internal://") {
      let relative = String(modelPath.dropFirst(11))
      return (getAppSupportDir() as NSString).appendingPathComponent(relative)
    }
    return (getAppSupportDir() as NSString).appendingPathComponent(modelPath)
  }

  // MARK: - Method handlers

  private func handleLoadModel(args: [String: Any], result: @escaping FlutterResult) {
    let modelPath = args["modelPath"] as? String ?? "yolo26n"
    let taskStr = args["task"] as? String ?? "detect"
    let instanceId = args["instanceId"] as? String ?? "default"
    let useGpu = args["useGpu"] as? Bool ?? true

    let resolvedPath = resolveModelPath(modelPath)

    if engines[instanceId] == nil {
      engines[instanceId] = YoloEngine()
    }

    do {
      try engines[instanceId]?.loadModel(path: resolvedPath, task: taskStr, useGpu: useGpu)
      result(true)
    } catch {
      result(FlutterError(code: "MODEL_NOT_FOUND", message: error.localizedDescription, details: nil))
    }
  }

  private func handlePredictSingleImage(args: [String: Any], result: @escaping FlutterResult) {
    guard let imageData = args["image"] as? FlutterStandardTypedData else {
      result(FlutterError(code: "bad_args", message: "No image data", details: nil))
      return
    }

    let instanceId = args["instanceId"] as? String ?? "default"
    let confidenceThreshold = args["confidenceThreshold"] as? Double ?? -1.0
    let iouThreshold = args["iouThreshold"] as? Double ?? -1.0

    guard let engine = engines[instanceId] else {
      result(FlutterError(code: "MODEL_NOT_LOADED", message: "Model not loaded", details: nil))
      return
    }

    do {
      let response = try engine.predict(
        imageData: imageData.data,
        confidenceThreshold: confidenceThreshold,
        iouThreshold: iouThreshold
      )
      result(response)
    } catch {
      result(FlutterError(code: "prediction_error", message: error.localizedDescription, details: nil))
    }
  }

  private func handleInspectModel(args: [String: Any], result: @escaping FlutterResult) {
    let modelPath = args["modelPath"] as? String ?? ""
    let resolvedPath = resolveModelPath(modelPath)

    do {
      let metadata = try ModelMetadata.inspect(path: resolvedPath)
      result(metadata)
    } catch {
      result(["path": resolvedPath, "task": "", "labels": []] as [String: Any])
    }
  }

  private func handleCheckModelExists(args: [String: Any], result: @escaping FlutterResult) {
    let modelPath = args["modelPath"] as? String ?? ""
    let resolvedPath = resolveModelPath(modelPath)
    let exists = FileManager.default.fileExists(atPath: resolvedPath)
    result(["exists": exists, "path": resolvedPath, "location": exists ? "local" : "not_found"] as [String: Any])
  }

  private func handleGetStoragePaths(result: @escaping FlutterResult) {
    let appSupport = getAppSupportDir()
    let cacheDir = (appSupport as NSString).appendingPathComponent("cache")
    try? FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
    result(["internal": appSupport, "cache": cacheDir, "external": "", "externalCache": ""] as [String: Any])
  }

  private func handleCreateInstance(args: [String: Any], result: @escaping FlutterResult) {
    guard let instanceId = args["instanceId"] as? String else {
      result(FlutterError(code: "bad_args", message: "Missing instanceId", details: nil))
      return
    }
    engines[instanceId] = nil
    result(nil)
  }

  private func handleDisposeInstance(args: [String: Any], result: @escaping FlutterResult) {
    let instanceId = args["instanceId"] as? String ?? "default"
    engines.removeValue(forKey: instanceId)
    result(nil)
  }

  private func handlePredictorInstance(args: [String: Any], result: @escaping FlutterResult) {
    let instanceId = args["instanceId"] as? String ?? "default"
    guard let engine = engines[instanceId] else {
      result(FlutterError(code: "predictor_instance_error", message: "Engine not initialized", details: nil))
      return
    }
    do {
      try engine.warmUp()
      result(nil)
    } catch {
      result(FlutterError(code: "predictor_instance_error", message: error.localizedDescription, details: nil))
    }
  }

  private func handleSetModel(args: [String: Any], result: @escaping FlutterResult) {
    let modelPath = args["modelPath"] as? String ?? ""
    let taskStr = args["task"] as? String ?? "detect"
    let useGpu = args["useGpu"] as? Bool ?? true

    let resolvedPath = resolveModelPath(modelPath)

    if engines["default"] == nil {
      engines["default"] = YoloEngine()
    }

    do {
      try engines["default"]?.loadModel(path: resolvedPath, task: taskStr, useGpu: useGpu)
      result(nil)
    } catch {
      result(FlutterError(code: "MODEL_NOT_FOUND", message: error.localizedDescription, details: nil))
    }
  }
}

// MARK: - YoloEngine (stub - replace with ONNX Runtime binding)

class YoloEngine {
  private var loaded = false
  private var task = "detect"

  func loadModel(path: String, task: String, useGpu: Bool) throws {
    guard FileManager.default.fileExists(atPath: path) else {
      throw NSError(domain: "YoloEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not found: \(path)"])
    }
    self.task = task
    self.loaded = true
    // TODO: Load ONNX Runtime session
  }

  func predict(imageData: Data, confidenceThreshold: Double, iouThreshold: Double) throws -> [String: Any] {
    guard loaded else {
      throw NSError(domain: "YoloEngine", code: 2, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
    }
    // TODO: Run inference with ONNX Runtime
    return ["boxes": [], "imageSize": ["width": 0, "height": 0], "speed": 0.0] as [String: Any]
  }

  func warmUp() throws {
    guard loaded else {
      throw NSError(domain: "YoloEngine", code: 2, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
    }
  }
}

// MARK: - ModelMetadata (stub)

class ModelMetadata {
  static func inspect(path: String) throws -> [String: Any] {
    return ["path": path, "task": "", "labels": []] as [String: Any]
  }
}
