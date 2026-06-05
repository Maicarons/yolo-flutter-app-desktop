// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:io' if (dart.library.js_interop) '';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart'
    if (dart.library.js_interop) '';
import 'package:ultralytics_yolo/config/channel_config.dart';
import 'package:ultralytics_yolo/core/yolo_model_manager.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/utils/mini_zip.dart'
    if (dart.library.js_interop) '';

class _OfficialModelArtifact {
  const _OfficialModelArtifact({
    required this.id,
    required this.task,
    this.assetName,
  });

  final String id;
  final YOLOTask task;

  /// Unified asset name for all desktop platforms (TFLite int8 format).
  /// Windows, Linux, and macOS all use the same TFLite models via ONNX Runtime.
  final String? assetName;
}

class YOLOResolvedModel {
  const YOLOResolvedModel({
    required this.modelPath,
    required this.task,
    required this.metadata,
  });

  final String modelPath;
  final YOLOTask task;
  final Map<String, dynamic> metadata;
}

class YOLOModelResolver {
  static const String _releaseBaseUrl =
      'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.3.5';

  static const List<String> _yolo26Sizes = ['n', 's', 'm', 'l', 'x'];
  static const List<({YOLOTask task, String suffix})> _yolo26Tasks = [
    (task: YOLOTask.detect, suffix: ''),
    (task: YOLOTask.segment, suffix: '-seg'),
    (task: YOLOTask.semantic, suffix: '-sem'),
    (task: YOLOTask.classify, suffix: '-cls'),
    (task: YOLOTask.pose, suffix: '-pose'),
    (task: YOLOTask.obb, suffix: '-obb'),
  ];

  static final List<_OfficialModelArtifact> _officialModels = [
    for (final task in _yolo26Tasks)
      for (final size in _yolo26Sizes)
        _yolo26Artifact(task: task.task, size: size, suffix: task.suffix),
    const _OfficialModelArtifact(
      id: 'yolo11n',
      task: YOLOTask.detect,
      assetName: 'yolo11n.tflite',
    ),
    const _OfficialModelArtifact(
      id: 'yolo11n-seg',
      task: YOLOTask.segment,
      assetName: 'yolo11n-seg.tflite',
    ),
    const _OfficialModelArtifact(
      id: 'yolo11n-cls',
      task: YOLOTask.classify,
      assetName: 'yolo11n-cls.tflite',
    ),
    const _OfficialModelArtifact(
      id: 'yolo11n-pose',
      task: YOLOTask.pose,
      assetName: 'yolo11n-pose.tflite',
    ),
    const _OfficialModelArtifact(
      id: 'yolo11n-obb',
      task: YOLOTask.obb,
      assetName: 'yolo11n-obb.tflite',
    ),
  ];

  static _OfficialModelArtifact _yolo26Artifact({
    required YOLOTask task,
    required String size,
    required String suffix,
  }) {
    final id = 'yolo26$size$suffix';
    return _OfficialModelArtifact(
      id: id,
      task: task,
      assetName: '${id}_int8.tflite',
    );
  }

  static List<String> officialModels({YOLOTask? task}) {
    return _officialModels
        .where((model) => task == null || model.task == task)
        .where(_isAvailableOnCurrentPlatform)
        .map((model) => model.id)
        .toList(growable: false);
  }

  static String? defaultOfficialModel({YOLOTask task = YOLOTask.detect}) {
    final models = officialModels(task: task);
    return models.isEmpty ? null : models.first;
  }

  static bool isOfficialModel(String source) =>
      _officialModelForId(_normalizeOfficialModelId(source)) != null;

  static bool _isAvailableOnCurrentPlatform(_OfficialModelArtifact model) {
    // All desktop platforms use the same TFLite models.
    return model.assetName != null;
  }

  static Future<YOLOResolvedModel> resolve({
    required String modelPath,
    YOLOTask? task,
  }) async {
    final resolvedPath = await preparePath(modelPath);
    final metadata = await inspect(resolvedPath);
    final metadataTask = YOLOTaskParsing.tryParse(metadata['task'] as String?);

    if (task != null && metadataTask != null && task != metadataTask) {
      throw ModelLoadingException(
        'Model task mismatch for $modelPath: expected ${task.name}, '
        'metadata says ${metadataTask.name}.',
      );
    }

    final effectiveTask = task ?? metadataTask;
    if (effectiveTask == null) {
      throw ModelLoadingException(
        'Could not determine the task for $modelPath. '
        'Provide task explicitly or use a model with exported metadata.',
      );
    }

    return YOLOResolvedModel(
      modelPath: resolvedPath,
      task: effectiveTask,
      metadata: metadata,
    );
  }

  static Future<Map<String, dynamic>> inspect(String modelPath) async {
    final channel = ChannelConfig.createSingleImageChannel();
    final result = await channel.invokeMethod('inspectModel', {
      'modelPath': modelPath,
    });
    if (result is! Map) return {};
    return Map<String, dynamic>.from(result);
  }

  static Future<String> preparePath(String modelPath) =>
      _resolvePath(modelPath);

  static Future<String> _resolvePath(String source) async {
    final officialId = _normalizeOfficialModelId(source);

    if (_officialModelForId(officialId) != null) {
      return _resolveOfficialModel(officialId!);
    }

    final uri = Uri.tryParse(source);
    if (uri != null && (uri.scheme == 'http' || uri.scheme == 'https')) {
      return _downloadRemoteModel(uri);
    }

    if (source.startsWith('assets/')) {
      return _copyFlutterAssetToAppSupport(source);
    }

    return source;
  }

  static String? _normalizeOfficialModelId(String source) {
    final fileName = source.split('/').last;
    final normalized = fileName
        .replaceAll('.mlpackage.zip', '')
        .replaceAll('.mlpackage', '')
        .replaceAll('.mlmodelc', '')
        .replaceAll('.mlmodel', '')
        .replaceAll('.tflite', '');
    return normalized.isEmpty ? null : normalized;
  }

  static _OfficialModelArtifact? _officialModelForId(String? modelId) {
    if (modelId == null) return null;
    for (final model in _officialModels) {
      if (model.id == modelId) return model;
    }
    return null;
  }

  static Future<String> _resolveOfficialModel(String modelId) async {
    final artifact = _officialModelForId(modelId);
    if (artifact == null) {
      throw ModelLoadingException('Unsupported official model: $modelId');
    }

    final filename = artifact.assetName;
    if (filename == null) {
      throw ModelLoadingException(
        'Official model ${artifact.id} is not available on this platform.',
      );
    }

    final directory = await getApplicationSupportDirectory();
    final modelFile = File('${directory.path}/$filename');
    if (modelFile.existsSync()) return modelFile.path;

    // Try bundled Flutter asset first.
    if (await _copyFlutterAssetIfExists('assets/models/$filename', modelFile)) {
      return modelFile.path;
    }

    // Download from release.
    await _downloadToFile(
      '$_releaseBaseUrl/$filename',
      modelFile,
      progressId: artifact.id,
    );
    return modelFile.path;
  }

  static Future<String> _downloadRemoteModel(Uri uri) async {
    final directory = await getApplicationSupportDirectory();
    final fileName =
        uri.pathSegments.isEmpty ? 'model' : uri.pathSegments.last;
    final file = File('${directory.path}/$fileName');
    if (file.existsSync()) return file.path;
    await _downloadToFile(
      uri.toString(),
      file,
      progressId: _normalizeOfficialModelId(fileName),
    );
    return file.path;
  }

  static Future<String> _copyFlutterAssetToAppSupport(String assetPath) async {
    final fileName = assetPath.split('/').last;
    final directory = await getApplicationSupportDirectory();
    final file = File('${directory.path}/$fileName');
    if (file.existsSync()) return file.path;

    final assetBytes = await _loadAssetBytes(assetPath);
    if (assetBytes == null) {
      throw ModelLoadingException('Flutter asset not found: $assetPath');
    }

    file.writeAsBytesSync(assetBytes, flush: true);
    return file.path;
  }

  static Future<bool> _copyFlutterAssetIfExists(
    String assetPath,
    File targetFile,
  ) async {
    final assetBytes = await _loadAssetBytes(assetPath);
    if (assetBytes == null) return false;
    targetFile.parent.createSync(recursive: true);
    targetFile.writeAsBytesSync(assetBytes, flush: true);
    return true;
  }

  static Future<List<int>?> _loadAssetBytes(String assetPath) async {
    try {
      final asset = await rootBundle.load(assetPath);
      return asset.buffer.asUint8List();
    } catch (_) {
      return null;
    }
  }

  static Future<void> _downloadToFile(
    String url,
    File targetFile, {
    String? progressId,
  }) async {
    targetFile.parent.createSync(recursive: true);
    final client = HttpClient();
    Object? downloadToken;
    if (progressId != null) {
      downloadToken = YOLOModelManager.registerDownload(
        progressId,
        () => client.close(force: true),
      );
    }
    final temporaryFile = File('${targetFile.path}.download');
    void checkCancelled() {
      if (progressId != null &&
          downloadToken != null &&
          YOLOModelManager.isDownloadCancelled(progressId, downloadToken)) {
        throw ModelLoadingException('Model download canceled.');
      }
    }

    try {
      if (temporaryFile.existsSync()) {
        temporaryFile.deleteSync();
      }
      checkCancelled();
      final request = await client.getUrl(Uri.parse(url));
      final response = await request.close();
      checkCancelled();
      if (response.statusCode != HttpStatus.ok) {
        throw ModelLoadingException(
          'Failed to download model from $url (HTTP ${response.statusCode}).',
        );
      }

      final totalBytes = response.contentLength;
      var receivedBytes = 0;
      double lastFraction = -1;

      if (progressId != null) {
        YOLOModelManager.emitProgress(progressId, 0);
      }

      final sink = temporaryFile.openWrite();
      try {
        await response.forEach((chunk) {
          checkCancelled();
          sink.add(chunk);
          if (progressId == null || totalBytes <= 0) return;
          receivedBytes += chunk.length;
          final fraction = (receivedBytes / totalBytes).clamp(0.0, 0.99);
          if (fraction - lastFraction >= 0.01) {
            lastFraction = fraction;
            YOLOModelManager.emitProgress(progressId, fraction);
          }
        });
      } finally {
        await sink.close();
      }

      checkCancelled();
      if (!temporaryFile.existsSync() || temporaryFile.lengthSync() == 0) {
        throw ModelLoadingException(
          'Downloaded 0 bytes for $url. The asset may be missing from the release.',
        );
      }

      if (targetFile.existsSync()) {
        targetFile.deleteSync();
      }
      temporaryFile.renameSync(targetFile.path);

      if (progressId != null) {
        YOLOModelManager.emitProgress(progressId, 1);
      }
    } catch (_) {
      if (temporaryFile.existsSync()) {
        temporaryFile.deleteSync();
      }
      if (progressId != null &&
          downloadToken != null &&
          YOLOModelManager.isDownloadCancelled(progressId, downloadToken)) {
        throw ModelLoadingException('Model download canceled.');
      }
      rethrow;
    } finally {
      if (progressId != null && downloadToken != null) {
        YOLOModelManager.finishDownload(progressId, downloadToken);
      }
      client.close(force: true);
    }
  }
}
