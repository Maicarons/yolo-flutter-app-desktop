// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef FLUTTER_PLUGIN_ULTRALYTICS_YOLO_MODEL_METADATA_H_
#define FLUTTER_PLUGIN_ULTRALYTICS_YOLO_MODEL_METADATA_H_

#include <flutter/encodable_value.h>

#include <string>
#include <vector>

namespace ultralytics_yolo {

/// Reads Ultralytics-appended ZIP metadata from a TFLite model file and
/// returns an EncodableMap with keys: task, labels, imgsz, stride, etc.
class ModelMetadata {
 public:
  /// Inspect a model file and return its metadata as a Flutter EncodableMap.
  /// Throws if the file cannot be read.
  static flutter::EncodableMap Inspect(const std::string& modelPath);

 private:
  /// Search for the ZIP end-of-central-directory signature in the last
  /// `searchBytes` of the file.  Returns the file offset or -1.
  static int64_t FindZipSignature(const std::vector<uint8_t>& data,
                                  size_t searchBytes);

  /// Parse a simple key-value YAML-like metadata block (Ultralytics format).
  static flutter::EncodableMap ParseYamlMetadata(const std::string& yaml);
};

}  // namespace ultralytics_yolo

#endif  // FLUTTER_PLUGIN_ULTRALYTICS_YOLO_MODEL_METADATA_H_
