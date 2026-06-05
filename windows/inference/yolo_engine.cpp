// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "yolo_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>

// ── stb_image (single-file JPEG/PNG decoder, embedded) ───────────────────────
// Vendored to avoid an external dependency.  Only the implementation is
// compiled in this translation unit.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace ultralytics_yolo {

// ── Lifecycle ────────────────────────────────────────────────────────────────

YoloEngine::YoloEngine()
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolo") {
  sessionOpts_.SetIntraOpNumThreads(4);
  sessionOpts_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
}

YoloEngine::~YoloEngine() = default;

// ── LoadModel ────────────────────────────────────────────────────────────────

void YoloEngine::LoadModel(const std::string& modelPath,
                           const std::string& task, bool useGpu) {
  std::wstring wModelPathCheck(modelPath.begin(), modelPath.end());
  if (!std::filesystem::exists(wModelPathCheck)) {
    throw std::runtime_error("Model file not found: " + modelPath);
  }

  task_ = task;

  // Try to load the model. ONNX Runtime supports ONNX natively.
  // TFLite support requires building ORT with TFLite delegate enabled.
  try {
    session_ = std::make_unique<Ort::Session>(env_, wModelPathCheck.c_str(),
                                              sessionOpts_);
  } catch (const Ort::Exception& e) {
    // Provide a clear error message for TFLite version issues.
    std::string msg = e.what();
    if (msg.find("version") != std::string::npos &&
        msg.find("not supported") != std::string::npos) {
      throw std::runtime_error(
          "TFLite model format not supported by this ONNX Runtime build. "
          "Model: " + modelPath + ". Error: " + msg +
          ". Consider converting the model to ONNX format.");
    }
    throw;
  }

  // Cache input/output names.
  inputNameStrings_.clear();
  outputNameStrings_.clear();
  inputNames_.clear();
  outputNames_.clear();

  size_t numInputs = session_->GetInputCount();
  for (size_t i = 0; i < numInputs; ++i) {
    auto name = session_->GetInputNameAllocated(i, allocator_);
    inputNameStrings_.push_back(name.get());
  }
  for (auto& s : inputNameStrings_) inputNames_.push_back(s.c_str());

  size_t numOutputs = session_->GetOutputCount();
  for (size_t i = 0; i < numOutputs; ++i) {
    auto name = session_->GetOutputNameAllocated(i, allocator_);
    outputNameStrings_.push_back(name.get());
  }
  for (auto& s : outputNameStrings_) outputNames_.push_back(s.c_str());

  // Infer input spatial dimensions from the tensor shape.
  if (numInputs > 0) {
    auto typeInfo = session_->GetInputTypeInfo(0);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    auto shape = tensorInfo.GetShape();  // e.g. [1, 640, 640, 3]
    if (shape.size() == 4) {
      // NHWC or NCHW — detect by looking at which axis is 3.
      if (shape[3] == 3) {
        inputH_ = static_cast<int>(shape[1]);
        inputW_ = static_cast<int>(shape[2]);
        inputChannels_ = 3;
      } else if (shape[1] == 3) {
        inputH_ = static_cast<int>(shape[2]);
        inputW_ = static_cast<int>(shape[3]);
        inputChannels_ = 3;
      } else {
        inputH_ = 640;
        inputW_ = 640;
        inputChannels_ = 3;
      }
    }
  }

  loaded_ = true;
}

// ── WarmUp ───────────────────────────────────────────────────────────────────

void YoloEngine::WarmUp() {
  if (!loaded_) return;
  // Create a zero-filled dummy input and run once to pre-allocate.
  size_t elemCount =
      static_cast<size_t>(1) * inputH_ * inputW_ * inputChannels_;
  std::vector<float> dummy(elemCount, 0.f);

  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> shape = {1, inputH_, inputW_, inputChannels_};
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, dummy.data(), dummy.size(), shape.data(), shape.size());

  session_->Run(Ort::RunOptions{nullptr}, inputNames_.data(),
                &inputTensor, 1, outputNames_.data(), outputNames_.size());
}

// ── Image decoding (stb_image) ───────────────────────────────────────────────

std::vector<uint8_t> YoloEngine::DecodeImage(
    const std::vector<uint8_t>& bytes, int& outW, int& outH,
    int& outChannels) {
  int w, h, c;
  unsigned char* pixels =
      stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                            &w, &h, &c, 3 /* force RGB */);
  if (!pixels) {
    throw std::runtime_error("Failed to decode image");
  }
  outW = w;
  outH = h;
  outChannels = 3;
  std::vector<uint8_t> rgb(pixels, pixels + w * h * 3);
  stbi_image_free(pixels);
  return rgb;
}

// ── Preprocess: letterbox resize + normalize to [0,1] ───────────────────────

std::vector<float> YoloEngine::Preprocess(const std::vector<uint8_t>& rgb,
                                          int srcW, int srcH, int srcC,
                                          int dstW, int dstH) {
  // Letterbox with gray fill (114).
  std::vector<float> output(dstW * dstH * 3, 114.f / 255.f);

  float scale = std::min(static_cast<float>(dstW) / srcW,
                         static_cast<float>(dstH) / srcH);
  int newW = static_cast<int>(srcW * scale);
  int newH = static_cast<int>(srcH * scale);
  int padX = (dstW - newW) / 2;
  int padY = (dstH - newH) / 2;

  for (int y = 0; y < newH; ++y) {
    int srcY = static_cast<int>(y / scale);
    if (srcY >= srcH) srcY = srcH - 1;
    for (int x = 0; x < newW; ++x) {
      int srcX = static_cast<int>(x / scale);
      if (srcX >= srcW) srcX = srcW - 1;

      int dstIdx = ((padY + y) * dstW + (padX + x)) * 3;
      int srcIdx = (srcY * srcW + srcX) * 3;
      output[dstIdx + 0] = rgb[srcIdx + 0] / 255.f;
      output[dstIdx + 1] = rgb[srcIdx + 1] / 255.f;
      output[dstIdx + 2] = rgb[srcIdx + 2] / 255.f;
    }
  }
  return output;
}

// ── NMS ──────────────────────────────────────────────────────────────────────

std::vector<int> YoloEngine::NMS(const std::vector<Detection>& dets,
                                 float iouThresh) {
  // Sort by confidence descending.
  std::vector<int> indices(dets.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return dets[a].confidence > dets[b].confidence; });

  std::vector<int> keep;
  std::vector<bool> suppressed(dets.size(), false);

  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    int idx = indices[i];
    if (suppressed[idx]) continue;
    keep.push_back(idx);

    for (int j = i + 1; j < static_cast<int>(indices.size()); ++j) {
      int jdx = indices[j];
      if (suppressed[jdx]) continue;

      // IoU.
      float x1 = std::max(dets[idx].x1, dets[jdx].x1);
      float y1 = std::max(dets[idx].y1, dets[jdx].y1);
      float x2 = std::min(dets[idx].x2, dets[jdx].x2);
      float y2 = std::min(dets[idx].y2, dets[jdx].y2);
      float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
      float areaA = (dets[idx].x2 - dets[idx].x1) * (dets[idx].y2 - dets[idx].y1);
      float areaB = (dets[jdx].x2 - dets[jdx].x1) * (dets[jdx].y2 - dets[jdx].y1);
      float unionArea = areaA + areaB - inter;
      if (unionArea > 0.f && inter / unionArea > iouThresh) {
        suppressed[jdx] = true;
      }
    }
  }
  return keep;
}

// ── PostProcess: Detect ──────────────────────────────────────────────────────

PredictResult YoloEngine::PostProcessDetect(
    const std::vector<float>& output, int outputLen, int imgW, int imgH,
    float confThresh, float iouThresh) {
  PredictResult pr;
  pr.imageWidth = imgW;
  pr.imageHeight = imgH;

  // Typical YOLO output shape: [1, numClasses+4, numDetections] or
  // [1, numDetections, numClasses+4].  We auto-detect.
  //
  // The int8 TFLite models exported by Ultralytics use [1, N, 4+numClasses]
  // layout where N = 8400 (for 640 input).

  int numDetections = 0;
  int stride = 0;

  // Try to infer layout from output size.
  // For 640 input, common N values: 8400, 2100, 525, ...
  // Common class counts: 80 (COCO), 1, ...
  // We look for the dimension that makes sense.
  if (outputLen > 100) {
    // Heuristic: if the first dimension is much larger, it's [1, N, stride].
    // For YOLO26 int8 TFLite: output shape is typically [1, 8400, 84].
    // We need the actual shape from the model.
    // For now, use a simple approach: try both layouts.
    //
    // Assume [1, N, stride] layout (most common for exported TFLite).
    // Find N by checking common values.
    int possibleN[] = {8400, 2100, 525, 132, 33, 25200, 6300, 1575};
    for (int n : possibleN) {
      if (n * (4 + 80) == outputLen || n * (4 + 1) == outputLen ||
          n * (4 + 80 + 32) == outputLen) {
        numDetections = n;
        stride = outputLen / n;
        break;
      }
    }

    if (numDetections == 0) {
      // Fallback: assume stride = 84 (4 + 80 COCO classes).
      stride = 84;
      numDetections = outputLen / stride;
    }
  }

  int numClasses = stride - 4;
  if (numClasses <= 0) numClasses = 80;

  std::vector<Detection> allDets;
  for (int i = 0; i < numDetections; ++i) {
    const float* row = output.data() + i * stride;

    // Find max class score.
    float maxClassScore = 0.f;
    int maxClassIdx = 0;
    for (int c = 0; c < numClasses; ++c) {
      if (row[4 + c] > maxClassScore) {
        maxClassScore = row[4 + c];
        maxClassIdx = c;
      }
    }

    float conf = maxClassScore;
    if (conf < confThresh) continue;

    // Box format: cx, cy, w, h (pixels).
    float cx = row[0], cy = row[1], w = row[2], h = row[3];
    Detection d;
    d.x1 = cx - w / 2;
    d.y1 = cy - h / 2;
    d.x2 = cx + w / 2;
    d.y2 = cy + h / 2;
    d.x1n = d.x1 / imgW;
    d.y1n = d.y1 / imgH;
    d.x2n = d.x2 / imgW;
    d.y2n = d.y2 / imgH;
    d.classIndex = maxClassIdx;
    d.className = std::to_string(maxClassIdx);  // Will be resolved by Dart
    d.confidence = conf;
    allDets.push_back(d);
  }

  auto keep = NMS(allDets, iouThresh);
  for (int idx : keep) {
    pr.boxes.push_back(allDets[idx]);
  }
  return pr;
}

// ── PostProcess: Segment ─────────────────────────────────────────────────────

PredictResult YoloEngine::PostProcessSegment(
    const std::vector<std::vector<float>>& outputs,
    const std::vector<std::vector<int64_t>>& outputShapes, int imgW, int imgH,
    float confThresh, float iouThresh) {
  // Segment models have 2 outputs: detection [1,N,84+32] and masks [1,32,160,160].
  PredictResult pr;
  if (outputs.size() < 2) {
    // Fallback: treat as detect.
    if (!outputs.empty()) {
      return PostProcessDetect(outputs[0],
                               static_cast<int>(outputs[0].size()), imgW, imgH,
                               confThresh, iouThresh);
    }
    return pr;
  }

  // First output: [1, N, 84+32] — use first 84 for detection.
  const auto& detOutput = outputs[0];
  int detLen = static_cast<int>(detOutput.size());
  int stride = 0;
  int numDetections = 0;

  // Try common N values.
  int possibleN[] = {8400, 2100, 525, 132, 33};
  for (int n : possibleN) {
    if (n > 0 && detLen % n == 0) {
      stride = detLen / n;
      if (stride >= 84) {
        numDetections = n;
        break;
      }
    }
  }
  if (numDetections == 0) {
    stride = 116;  // 84 + 32
    numDetections = detLen / stride;
  }

  int numClasses = 80;
  int protoDim = stride - 4 - numClasses;  // typically 32

  // Detect boxes.
  std::vector<Detection> allDets;
  std::vector<std::vector<float>> maskCoeffs;

  for (int i = 0; i < numDetections; ++i) {
    const float* row = detOutput.data() + i * stride;

    float maxClassScore = 0.f;
    int maxClassIdx = 0;
    for (int c = 0; c < numClasses; ++c) {
      if (row[4 + c] > maxClassScore) {
        maxClassScore = row[4 + c];
        maxClassIdx = c;
      }
    }
    if (maxClassScore < confThresh) continue;

    float cx = row[0], cy = row[1], w = row[2], h = row[3];
    Detection d;
    d.x1 = cx - w / 2;
    d.y1 = cy - h / 2;
    d.x2 = cx + w / 2;
    d.y2 = cy + h / 2;
    d.x1n = d.x1 / imgW;
    d.y1n = d.y1 / imgH;
    d.x2n = d.x2 / imgW;
    d.y2n = d.y2 / imgH;
    d.classIndex = maxClassIdx;
    d.className = std::to_string(maxClassIdx);
    d.confidence = maxClassScore;
    allDets.push_back(d);

    // Extract mask coefficients.
    std::vector<float> coeffs(row + 4 + numClasses,
                              row + 4 + numClasses + protoDim);
    maskCoeffs.push_back(coeffs);
  }

  auto keep = NMS(allDets, iouThresh);

  // Second output: prototype masks [1, protoDim, maskH, maskW].
  const auto& protoOutput = outputs[1];
  int maskH = 160, maskW = 160;
  if (outputShapes.size() > 1 && outputShapes[1].size() == 4) {
    maskH = static_cast<int>(outputShapes[1][2]);
    maskW = static_cast<int>(outputShapes[1][3]);
  }
  int protoDimActual = protoDim;
  if (outputShapes.size() > 1 && outputShapes[1].size() >= 2) {
    protoDimActual = static_cast<int>(outputShapes[1][1]);
  }

  for (int idx : keep) {
    pr.boxes.push_back(allDets[idx]);

    // Generate instance mask: coefficients × prototypes.
    const auto& coeffs = maskCoeffs[idx];
    std::vector<std::vector<double>> mask(maskH,
                                          std::vector<double>(maskW, 0.0));

    for (int y = 0; y < maskH; ++y) {
      for (int x = 0; x < maskW; ++x) {
        float val = 0.f;
        for (int d = 0; d < std::min(static_cast<int>(coeffs.size()),
                                      protoDimActual);
             ++d) {
          // proto layout: [protoDim, maskH, maskW] → index = d*maskH*maskW + y*maskW + x
          int protoIdx = d * maskH * maskW + y * maskW + x;
          if (protoIdx < static_cast<int>(protoOutput.size())) {
            val += coeffs[d] * protoOutput[protoIdx];
          }
        }
        mask[y][x] = 1.0 / (1.0 + std::exp(-val));  // sigmoid
      }
    }
    pr.masks.push_back(mask);
  }

  pr.imageWidth = imgW;
  pr.imageHeight = imgH;
  return pr;
}

// ── PostProcess: Classify ────────────────────────────────────────────────────

PredictResult YoloEngine::PostProcessClassify(const std::vector<float>& output,
                                              int numClasses) {
  PredictResult pr;

  // Softmax.
  float maxVal = *std::max_element(output.begin(), output.end());
  std::vector<float> probs(numClasses);
  float sum = 0.f;
  for (int i = 0; i < numClasses; ++i) {
    probs[i] = std::exp(output[i] - maxVal);
    sum += probs[i];
  }
  for (int i = 0; i < numClasses; ++i) probs[i] /= sum;

  // Top-1.
  int top1Idx = static_cast<int>(
      std::max_element(probs.begin(), probs.end()) - probs.begin());
  pr.clsTop1Index = top1Idx;
  pr.clsTop1Label = std::to_string(top1Idx);
  pr.clsTop1Conf = probs[top1Idx];

  // Top-5.
  std::vector<int> indices(numClasses);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return probs[a] > probs[b]; });
  for (int i = 0; i < std::min(5, numClasses); ++i) {
    pr.clsTop5.push_back({indices[i], std::to_string(indices[i]), probs[indices[i]]});
  }

  pr.imageWidth = 1;
  pr.imageHeight = 1;
  return pr;
}

// ── PostProcess: Pose ────────────────────────────────────────────────────────

PredictResult YoloEngine::PostProcessPose(
    const std::vector<float>& output, int outputLen, int imgW, int imgH,
    float confThresh, float iouThresh) {
  // Pose output: [1, N, 4 + numClasses + numKeypoints*3]
  // For COCO pose: 17 keypoints × 3 (x,y,conf) = 51 extra dims.
  PredictResult pr;
  pr.imageWidth = imgW;
  pr.imageHeight = imgH;

  int possibleN[] = {8400, 2100, 525, 132, 33};
  int numDetections = 0, stride = 0;
  for (int n : possibleN) {
    if (n > 0 && outputLen % n == 0) {
      stride = outputLen / n;
      if (stride >= 56) {  // 4 + 1 + 51 minimum
        numDetections = n;
        break;
      }
    }
  }
  if (numDetections == 0) {
    stride = 56;
    numDetections = outputLen / stride;
  }

  int numClasses = 1;
  int kpStride = stride - 4 - numClasses;  // 51 for COCO
  int numKeypoints = kpStride / 3;

  std::vector<Detection> allDets;
  std::vector<std::vector<PredictResult::KeypointCoord>> allKeypoints;

  for (int i = 0; i < numDetections; ++i) {
    const float* row = output.data() + i * stride;
    float objConf = row[4];  // single-class confidence
    if (objConf < confThresh) continue;

    float cx = row[0], cy = row[1], w = row[2], h = row[3];
    Detection d;
    d.x1 = cx - w / 2;
    d.y1 = cy - h / 2;
    d.x2 = cx + w / 2;
    d.y2 = cy + h / 2;
    d.x1n = d.x1 / imgW;
    d.y1n = d.y1 / imgH;
    d.x2n = d.x2 / imgW;
    d.y2n = d.y2 / imgH;
    d.classIndex = 0;
    d.className = "0";
    d.confidence = objConf;
    allDets.push_back(d);

    // Keypoints.
    std::vector<PredictResult::KeypointCoord> kps;
    for (int k = 0; k < numKeypoints; ++k) {
      const float* kp = row + 4 + numClasses + k * 3;
      kps.push_back({kp[0], kp[1], kp[2]});
    }
    allKeypoints.push_back(kps);
  }

  auto keep = NMS(allDets, iouThresh);
  for (int idx : keep) {
    pr.boxes.push_back(allDets[idx]);
    pr.keypoints.push_back(allKeypoints[idx]);
  }
  return pr;
}

// ── PostProcess: OBB ─────────────────────────────────────────────────────────

PredictResult YoloEngine::PostProcessOBB(
    const std::vector<float>& output, int outputLen, int imgW, int imgH,
    float confThresh, float iouThresh) {
  // OBB output: [1, N, 4 + numClasses + 1(angle)]
  PredictResult pr;
  pr.imageWidth = imgW;
  pr.imageHeight = imgH;

  int possibleN[] = {8400, 2100, 525, 132, 33};
  int numDetections = 0, stride = 0;
  for (int n : possibleN) {
    if (n > 0 && outputLen % n == 0) {
      stride = outputLen / n;
      if (stride >= 5) {
        numDetections = n;
        break;
      }
    }
  }
  if (numDetections == 0) {
    stride = 85;  // 4 + 80 + 1
    numDetections = outputLen / stride;
  }

  int numClasses = stride - 5;  // subtract 4 box + 1 angle

  for (int i = 0; i < numDetections; ++i) {
    const float* row = output.data() + i * stride;

    float maxClassScore = 0.f;
    int maxClassIdx = 0;
    for (int c = 0; c < numClasses; ++c) {
      if (row[4 + c] > maxClassScore) {
        maxClassScore = row[4 + c];
        maxClassIdx = c;
      }
    }
    if (maxClassScore < confThresh) continue;

    float cx = row[0], cy = row[1], w = row[2], h = row[3];
    float angle = row[4 + numClasses];

    // Convert rotated box to 4-corner polygon.
    float cosA = std::cos(angle), sinA = std::sin(angle);
    float hw = w / 2, hh = h / 2;
    PredictResult::OBBResult obb;
    obb.polygon = {
        {cx + cosA * hw - sinA * hh, cy + sinA * hw + cosA * hh},
        {cx - cosA * hw - sinA * hh, cy - sinA * hw + cosA * hh},
        {cx - cosA * hw + sinA * hh, cy - sinA * hw - cosA * hh},
        {cx + cosA * hw + sinA * hh, cy + sinA * hw - cosA * hh},
    };
    obb.angle = angle;
    obb.classIndex = maxClassIdx;
    obb.className = std::to_string(maxClassIdx);
    obb.confidence = maxClassScore;
    pr.obb.push_back(obb);
  }

  return pr;
}

// ── ToEncodableMap ───────────────────────────────────────────────────────────

flutter::EncodableMap YoloEngine::ToEncodableMap(const PredictResult& pr) {
  flutter::EncodableMap response;

  // boxes
  flutter::EncodableList boxList;
  for (const auto& d : pr.boxes) {
    flutter::EncodableMap box;
    box[flutter::EncodableValue("x1")] = static_cast<double>(d.x1);
    box[flutter::EncodableValue("y1")] = static_cast<double>(d.y1);
    box[flutter::EncodableValue("x2")] = static_cast<double>(d.x2);
    box[flutter::EncodableValue("y2")] = static_cast<double>(d.y2);
    box[flutter::EncodableValue("x1_norm")] = static_cast<double>(d.x1n);
    box[flutter::EncodableValue("y1_norm")] = static_cast<double>(d.y1n);
    box[flutter::EncodableValue("x2_norm")] = static_cast<double>(d.x2n);
    box[flutter::EncodableValue("y2_norm")] = static_cast<double>(d.y2n);
    box[flutter::EncodableValue("class")] = d.className;
    box[flutter::EncodableValue("className")] = d.className;
    box[flutter::EncodableValue("confidence")] = static_cast<double>(d.confidence);
    boxList.push_back(box);
  }
  response[flutter::EncodableValue("boxes")] = boxList;

  // imageSize
  flutter::EncodableMap sizeMap;
  sizeMap[flutter::EncodableValue("width")] = pr.imageWidth;
  sizeMap[flutter::EncodableValue("height")] = pr.imageHeight;
  response[flutter::EncodableValue("imageSize")] = sizeMap;

  // speed
  response[flutter::EncodableValue("speed")] = static_cast<double>(pr.speedMs);

  // Task-specific fields.
  if (task_ == "segment" && !pr.masks.empty()) {
    flutter::EncodableList maskList;
    for (const auto& instanceMask : pr.masks) {
      flutter::EncodableList rows;
      for (const auto& row : instanceMask) {
        flutter::EncodableList rowData;
        for (double v : row) rowData.push_back(v);
        rows.push_back(rowData);
      }
      maskList.push_back(rows);
    }
    response[flutter::EncodableValue("masks")] = maskList;
  }

  if (task_ == "semantic" && !pr.semanticClassMap.empty()) {
    flutter::EncodableMap semMap;
    flutter::EncodableList classMapList;
    for (int64_t v : pr.semanticClassMap) classMapList.push_back(v);
    semMap[flutter::EncodableValue("classMap")] = classMapList;
    semMap[flutter::EncodableValue("width")] = pr.semanticWidth;
    semMap[flutter::EncodableValue("height")] = pr.semanticHeight;
    response[flutter::EncodableValue("semanticMask")] = semMap;
  }

  if (task_ == "classify") {
    flutter::EncodableMap clsMap;
    clsMap[flutter::EncodableValue("name")] = pr.clsTop1Label;
    clsMap[flutter::EncodableValue("class")] = pr.clsTop1Index;
    clsMap[flutter::EncodableValue("confidence")] =
        static_cast<double>(pr.clsTop1Conf);

    flutter::EncodableList top5;
    for (const auto& t : pr.clsTop5) {
      flutter::EncodableMap m;
      m[flutter::EncodableValue("class")] = t.classIdx;
      m[flutter::EncodableValue("name")] = t.name;
      m[flutter::EncodableValue("confidence")] = static_cast<double>(t.conf);
      top5.push_back(m);
    }
    clsMap[flutter::EncodableValue("top5")] = top5;
    response[flutter::EncodableValue("classification")] = clsMap;
  }

  if (task_ == "pose" && !pr.keypoints.empty()) {
    flutter::EncodableList kpList;
    for (const auto& personKps : pr.keypoints) {
      flutter::EncodableMap person;
      flutter::EncodableList coords;
      for (const auto& kp : personKps) {
        flutter::EncodableMap c;
        c[flutter::EncodableValue("x")] = static_cast<double>(kp.x);
        c[flutter::EncodableValue("y")] = static_cast<double>(kp.y);
        c[flutter::EncodableValue("confidence")] = static_cast<double>(kp.conf);
        coords.push_back(c);
      }
      person[flutter::EncodableValue("coordinates")] = coords;
      kpList.push_back(person);
    }
    response[flutter::EncodableValue("keypoints")] = kpList;
  }

  if (task_ == "obb" && !pr.obb.empty()) {
    flutter::EncodableList obbList;
    for (const auto& o : pr.obb) {
      flutter::EncodableMap om;
      flutter::EncodableList pts;
      for (const auto& [px, py] : o.polygon) {
        flutter::EncodableMap p;
        p[flutter::EncodableValue("x")] = static_cast<double>(px);
        p[flutter::EncodableValue("y")] = static_cast<double>(py);
        pts.push_back(p);
      }
      om[flutter::EncodableValue("points")] = pts;
      om[flutter::EncodableValue("angle")] = static_cast<double>(o.angle);
      om[flutter::EncodableValue("classIndex")] = o.classIndex;
      om[flutter::EncodableValue("class")] = o.className;
      om[flutter::EncodableValue("confidence")] =
          static_cast<double>(o.confidence);
      obbList.push_back(om);
    }
    response[flutter::EncodableValue("obb")] = obbList;
  }

  return response;
}

// ── Predict ──────────────────────────────────────────────────────────────────

flutter::EncodableMap YoloEngine::Predict(
    const std::vector<uint8_t>& imageBytes, double confThreshold,
    double iouThreshold) {
  if (!loaded_) throw std::runtime_error("Model not loaded");

  float confThresh = confThreshold >= 0 ? static_cast<float>(confThreshold)
                                        : 0.25f;
  float iouThresh =
      iouThreshold >= 0 ? static_cast<float>(iouThreshold) : 0.7f;

  // Decode image.
  int imgW, imgH, imgC;
  auto rgb = DecodeImage(imageBytes, imgW, imgH, imgC);

  // Preprocess.
  auto inputTensor = Preprocess(rgb, imgW, imgH, imgC, inputW_, inputH_);

  // Run inference.
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> inputShape = {1, inputH_, inputW_, inputChannels_};
  Ort::Value inputOrtTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensor.data(), inputTensor.size(), inputShape.data(),
      inputShape.size());

  auto startTime = std::chrono::high_resolution_clock::now();

  auto outputTensors = session_->Run(
      Ort::RunOptions{nullptr}, inputNames_.data(), &inputOrtTensor, 1,
      outputNames_.data(), outputNames_.size());

  auto endTime = std::chrono::high_resolution_clock::now();
  float speedMs = std::chrono::duration<float, std::milli>(endTime - startTime)
                      .count();

  // Extract output data.
  std::vector<std::vector<float>> outputs;
  std::vector<std::vector<int64_t>> outputShapes;
  for (auto& tensor : outputTensors) {
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    outputShapes.push_back(shape);
    const float* data = tensor.GetTensorData<float>();
    size_t len = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    outputs.push_back(std::vector<float>(data, data + len));
  }

  // Post-process based on task.
  PredictResult pr;
  if (outputs.empty()) throw std::runtime_error("No output from model");

  if (task_ == "detect") {
    pr = PostProcessDetect(outputs[0], static_cast<int>(outputs[0].size()),
                           imgW, imgH, confThresh, iouThresh);
  } else if (task_ == "segment") {
    pr = PostProcessSegment(outputs, outputShapes, imgW, imgH, confThresh,
                            iouThresh);
  } else if (task_ == "classify") {
    pr = PostProcessClassify(outputs[0],
                             static_cast<int>(outputs[0].size()));
  } else if (task_ == "pose") {
    pr = PostProcessPose(outputs[0], static_cast<int>(outputs[0].size()),
                         imgW, imgH, confThresh, iouThresh);
  } else if (task_ == "obb") {
    pr = PostProcessOBB(outputs[0], static_cast<int>(outputs[0].size()),
                        imgW, imgH, confThresh, iouThresh);
  } else {
    // Default to detect.
    pr = PostProcessDetect(outputs[0], static_cast<int>(outputs[0].size()),
                           imgW, imgH, confThresh, iouThresh);
  }

  pr.speedMs = speedMs;
  return ToEncodableMap(pr);
}

}  // namespace ultralytics_yolo
