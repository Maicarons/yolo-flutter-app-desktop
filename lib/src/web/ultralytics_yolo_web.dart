// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_web_plugins/flutter_web_plugins.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';

/// Web implementation of the YOLO plugin using ONNX Runtime Web.
class YoloPluginWeb extends Plugin {
  YoloPluginWeb();

  static void registerWith(Registrar registrar) {
    final channel = MethodChannel(
      'yolo_single_image_channel',
      const StandardMethodCodec(),
      registrar,
    );
    final instance = YoloPluginWeb();
    channel.setMethodCallHandler(instance.handleMethodCall);
  }

  Future<dynamic> handleMethodCall(MethodCall call) async {
    switch (call.method) {
      case 'loadModel':
        return _handleLoadModel(call.arguments);
      case 'predictSingleImage':
        return _handlePredictSingleImage(call.arguments);
      case 'inspectModel':
        return _handleInspectModel(call.arguments);
      case 'checkModelExists':
        return _handleCheckModelExists(call.arguments);
      case 'getStoragePaths':
        return _handleGetStoragePaths();
      case 'createInstance':
        return _handleCreateInstance(call.arguments);
      case 'disposeInstance':
        return _handleDisposeInstance(call.arguments);
      case 'predictorInstance':
        return _handlePredictorInstance(call.arguments);
      case 'setModel':
        return _handleSetModel(call.arguments);
      case 'getPlatformVersion':
        return 'Web';
      default:
        throw MissingPluginException('Unknown method: ${call.method}');
    }
  }

  // ── Model management ───────────────────────────────────────────────────

  final Map<String, _WebYoloEngine> _engines = {};

  dynamic _handleLoadModel(dynamic arguments) {
    final args = arguments as Map;
    final modelPath = args['modelPath'] as String? ?? 'yolo26n';
    final task = args['task'] as String? ?? 'detect';
    final instanceId = args['instanceId'] as String? ?? 'default';

    if (_engines[instanceId] == null) {
      _engines[instanceId] = _WebYoloEngine();
    }

    try {
      _engines[instanceId]!.loadModel(modelPath, task);
      return true;
    } catch (e) {
      throw PlatformException(
        code: 'MODEL_NOT_FOUND',
        message: 'Failed to load model: $e',
      );
    }
  }

  dynamic _handlePredictSingleImage(dynamic arguments) {
    final args = arguments as Map;
    final imageData = args['image'] as Uint8List?;
    final instanceId = args['instanceId'] as String? ?? 'default';

    if (imageData == null || imageData.isEmpty) {
      throw PlatformException(code: 'bad_args', message: 'No image data');
    }

    final engine = _engines[instanceId];
    if (engine == null) {
      throw PlatformException(
        code: 'MODEL_NOT_LOADED',
        message: 'Model not loaded. Call loadModel() first.',
      );
    }

    try {
      return engine.predict(imageData);
    } catch (e) {
      throw PlatformException(
        code: 'prediction_error',
        message: 'Prediction failed: $e',
      );
    }
  }

  dynamic _handleInspectModel(dynamic arguments) {
    final args = arguments as Map;
    final modelPath = args['modelPath'] as String? ?? '';
    return {'path': modelPath, 'task': '', 'labels': <String>[]};
  }

  dynamic _handleCheckModelExists(dynamic arguments) {
    final args = arguments as Map;
    final modelPath = args['modelPath'] as String? ?? '';
    // On web, we can't check filesystem. Assume remote models exist.
    return {
      'exists': modelPath.startsWith('http'),
      'path': modelPath,
      'location': 'remote',
    };
  }

  dynamic _handleGetStoragePaths() {
    return {
      'internal': '/web',
      'cache': '/web/cache',
      'external': '',
      'externalCache': '',
    };
  }

  dynamic _handleCreateInstance(dynamic arguments) {
    final args = arguments as Map;
    final instanceId = args['instanceId'] as String?;
    if (instanceId == null) {
      throw PlatformException(code: 'bad_args', message: 'Missing instanceId');
    }
    _engines[instanceId] = _WebYoloEngine();
    return null;
  }

  dynamic _handleDisposeInstance(dynamic arguments) {
    final args = arguments as Map;
    final instanceId = args['instanceId'] as String? ?? 'default';
    _engines.remove(instanceId);
    return null;
  }

  dynamic _handlePredictorInstance(dynamic arguments) {
    // No-op on web - models are loaded on demand.
    return null;
  }

  dynamic _handleSetModel(dynamic arguments) {
    final args = arguments as Map;
    final modelPath = args['modelPath'] as String? ?? '';
    final task = args['task'] as String? ?? 'detect';

    if (_engines['default'] == null) {
      _engines['default'] = _WebYoloEngine();
    }

    try {
      _engines['default']!.loadModel(modelPath, task);
      return null;
    } catch (e) {
      throw PlatformException(
        code: 'MODEL_NOT_FOUND',
        message: 'Failed to load model: $e',
      );
    }
  }
}

/// Web-based YOLO engine using ONNX Runtime Web.
///
/// This is a stub implementation. The actual ONNX Runtime Web integration
/// requires loading the onnxruntime-web JavaScript library and running
/// inference in the browser.
class _WebYoloEngine {
  bool _loaded = false;
  String _task = 'detect';

  void loadModel(String modelPath, String task) {
    _task = task;
    _loaded = true;
    // TODO: Load ONNX model using ONNX Runtime Web JS API
    // Example:
    //   import 'package:web/web.dart' as web;
    //   final session = await web.window.callMethod('ort.InferenceSession.create', [modelPath]);
  }

  Map<String, dynamic> predict(Uint8List imageData) {
    if (!_loaded) {
      throw StateError('Model not loaded');
    }
    // TODO: Run inference with ONNX Runtime Web
    // This requires:
    // 1. Decode image to tensor
    // 2. Run inference via JS interop
    // 3. Parse results
    return {
      'boxes': <Map<String, dynamic>>[],
      'imageSize': {'width': 0, 'height': 0},
      'speed': 0.0,
    };
  }
}
