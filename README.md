# 🚀 YOLO Flutter Desktop

Ultralytics YOLO Flutter plugin for **desktop platforms** (Windows, Linux, macOS, Web).

This is a fork of [ultralytics/yolo-flutter-app](https://github.com/ultralytics/yolo-flutter-app) adapted for desktop platforms using **ONNX Runtime** as the inference backend.

## ✨ Supported Platforms

| Platform | Inference Engine | Status |
|----------|-----------------|--------|
| Windows  | ONNX Runtime C++ | ✅ Working |
| Linux    | ONNX Runtime C++ | ✅ Working |
| macOS    | ONNX Runtime C++ | 🚧 Stub (needs ONNX Runtime binding) |
| Web      | ONNX Runtime Web | 🚧 Stub (needs JS interop) |

## ⚡ Quick Start

```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

// Single image inference
final yolo = YOLO(modelPath: 'yolo26n');
await yolo.loadModel();
final results = await yolo.predict(imageBytes);

// Real-time camera inference
YOLOView(
  modelPath: 'yolo26n',
  onResult: (results) {
    for (final r in results) {
      print('${r.className}: ${r.confidence}');
    }
  },
)
```

## 📦 Model Assets

All desktop platforms use the same **TFLite int8** models from the [v0.3.5 release](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.3.5).

Models are automatically downloaded on first use and cached in the app's support directory.

## 🔨 Building

### Windows
```bash
flutter build windows
```

### Linux
```bash
flutter build linux
```

### macOS
```bash
flutter build macos
```

### Web
```bash
flutter build web
```

## 🎯 Features

- **Single Image Inference**: `YOLO` class for one-shot predictions
- **Real-time Camera Inference**: `YOLOView` widget for live camera feeds
- **6 Task Types**: Detection, Segmentation, Semantic Segmentation, Classification, Pose Estimation, OBB
- **5 Model Sizes**: Nano, Small, Medium, Large, XLarge
- **Auto Model Download**: Models are downloaded and cached automatically
- **Multi-instance Support**: Run multiple YOLO instances simultaneously

## 📚 Documentation

See the [original documentation](https://github.com/ultralytics/yolo-flutter-app) for detailed API usage.

## 📄 License

AGPL-3.0 License - See [LICENSE](LICENSE) for details.
