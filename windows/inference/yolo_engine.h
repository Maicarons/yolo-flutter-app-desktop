// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef FLUTTER_PLUGIN_ULTRALYTICS_YOLO_ENGINE_H_
#define FLUTTER_PLUGIN_ULTRALYTICS_YOLO_ENGINE_H_

#include <onnxruntime_cxx_api.h>

#include <flutter/encodable_value.h>

#include <memory>
#include <string>
#include <vector>

namespace ultralytics_yolo {

struct Detection {
  float x1, y1, x2, y2;      // pixel coords
  float x1n, y1n, x2n, y2n;  // normalized [0,1]
  int classIndex;
  std::string className;
  float confidence;
};

// Holds task-specific outputs that the caller (yolo_plugin) folds into the
// method-channel response map.
struct PredictResult {
  std::vector<Detection> boxes;
  int imageWidth = 0;
  int imageHeight = 0;
  float speedMs = 0.f;

  // segmentation: per-instance masks as 2-D float grids.
  std::vector<std::vector<std::vector<double>>> masks;

  // semantic segmentation: class-index map (H×W).
  std::vector<int64_t> semanticClassMap;
  int semanticWidth = 0;
  int semanticHeight = 0;

  // classification: top-1 + top-5.
  int clsTop1Index = 0;
  std::string clsTop1Label;
  float clsTop1Conf = 0.f;
  struct ClsTop5 {
    int classIdx;
    std::string name;
    float conf;
  };
  std::vector<ClsTop5> clsTop5;

  // pose: per-box keypoints  (x, y, confidence) triples.
  struct KeypointCoord {
    float x, y, conf;
  };
  std::vector<std::vector<KeypointCoord>> keypoints;

  // OBB: oriented bounding boxes.
  struct OBBResult {
    std::vector<std::pair<float, float>> polygon;  // 4 corners
    float angle = 0.f;
    int classIndex = 0;
    std::string className;
    float confidence = 0.f;
  };
  std::vector<OBBResult> obb;
};

class YoloEngine {
 public:
  YoloEngine();
  ~YoloEngine();

  // Load a TFLite or ONNX model.  `task` is one of
  // "detect","segment","semantic","classify","pose","obb".
  void LoadModel(const std::string& modelPath, const std::string& task,
                 bool useGpu = true);

  // Run inference on raw image bytes (JPEG/PNG).  Returns structured results.
  // `confThreshold` < 0 means "use default" (0.25).
  flutter::EncodableMap Predict(const std::vector<uint8_t>& imageBytes,
                                double confThreshold = -1.0,
                                double iouThreshold = -1.0);

  // Pre-allocate tensors with a dummy forward pass.
  void WarmUp();

  bool IsLoaded() const { return loaded_; }

 private:
  // Decode JPEG/PNG bytes into RGB pixel buffer.
  std::vector<uint8_t> DecodeImage(const std::vector<uint8_t>& bytes,
                                   int& outWidth, int& outHeight,
                                   int& outChannels);

  // Letterbox-resize to model input size.
  std::vector<float> Preprocess(const std::vector<uint8_t>& rgb, int srcW,
                                int srcH, int srcC, int dstW, int dstH);

  // Post-processing per task.
  PredictResult PostProcessDetect(const std::vector<float>& output,
                                  int outputLen, int imgW, int imgH,
                                  float confThresh, float iouThresh);

  PredictResult PostProcessSegment(const std::vector<std::vector<float>>& outputs,
                                   const std::vector<std::vector<int64_t>>& outputShapes,
                                   int imgW, int imgH, float confThresh,
                                   float iouThresh);

  PredictResult PostProcessClassify(const std::vector<float>& output,
                                    int numClasses);

  PredictResult PostProcessPose(const std::vector<float>& output,
                                int outputLen, int imgW, int imgH,
                                float confThresh, float iouThresh);

  PredictResult PostProcessOBB(const std::vector<float>& output,
                               int outputLen, int imgW, int imgH,
                               float confThresh, float iouThresh);

  // NMS helper.
  std::vector<int> NMS(const std::vector<Detection>& dets, float iouThresh);

  // Convert result to Flutter EncodableMap.
  flutter::EncodableMap ToEncodableMap(const PredictResult& pr);

  // ── Members ───────────────────────────────────────────────────────────
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "yolo"};
  Ort::SessionOptions sessionOpts_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::string task_ = "detect";
  int inputW_ = 640;
  int inputH_ = 640;
  int inputChannels_ = 3;
  bool loaded_ = false;

  // Cached input/output names (owned by the session).
  std::vector<std::string> inputNameStrings_;
  std::vector<std::string> outputNameStrings_;
  std::vector<const char*> inputNames_;
  std::vector<const char*> outputNames_;
};

}  // namespace ultralytics_yolo

#endif  // FLUTTER_PLUGIN_ULTRALYTICS_YOLO_ENGINE_H_
