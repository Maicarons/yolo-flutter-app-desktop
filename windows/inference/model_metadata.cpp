// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "model_metadata.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ultralytics_yolo {

// ZIP end-of-central-directory signature: 0x06054b50
static constexpr uint32_t kZipEndSig = 0x06054b50;
// ZIP central-directory-file-header signature: 0x02014b50
static constexpr uint32_t kZipCdhSig = 0x02014b50;
// Maximum tail scan (Ultralytics metadata is typically < 64 KB).
static constexpr size_t kMaxSearchBytes = 512 * 1024;

// ── Helpers ──────────────────────────────────────────────────────────────────

static uint32_t ReadU32LE(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

static uint16_t ReadU16LE(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

// ── FindZipSignature ─────────────────────────────────────────────────────────

int64_t ModelMetadata::FindZipSignature(const std::vector<uint8_t>& data,
                                        size_t searchBytes) {
  if (data.size() < 4) return -1;
  size_t start = data.size() > searchBytes ? data.size() - searchBytes : 0;
  for (size_t i = data.size() - 4; i >= start; --i) {
    if (ReadU32LE(data.data() + i) == kZipEndSig) {
      return static_cast<int64_t>(i);
    }
    if (i == 0) break;  // prevent underflow
  }
  return -1;
}

// ── ParseYamlMetadata (simplified) ───────────────────────────────────────────

flutter::EncodableMap ModelMetadata::ParseYamlMetadata(
    const std::string& yaml) {
  flutter::EncodableMap map;
  std::istringstream stream(yaml);
  std::string line;

  while (std::getline(stream, line)) {
    // Skip empty lines and comments.
    if (line.empty() || line[0] == '#') continue;

    auto colonPos = line.find(':');
    if (colonPos == std::string::npos) continue;

    std::string key = line.substr(0, colonPos);
    std::string value = line.substr(colonPos + 1);

    // Trim whitespace.
    auto trim = [](std::string& s) {
      s.erase(0, s.find_first_not_of(" \t\r\n"));
      s.erase(s.find_last_not_of(" \t\r\n") + 1);
    };
    trim(key);
    trim(value);

    if (key == "task") {
      map[flutter::EncodableValue("task")] = value;
    } else if (key == "imgsz") {
      // Could be "[640, 640]" or "640".
      if (!value.empty() && value[0] == '[') {
        // Parse list — just store as string for now.
        map[flutter::EncodableValue("imgsz")] = value;
      } else {
        try {
          map[flutter::EncodableValue("imgsz")] = std::stoi(value);
        } catch (...) {
          map[flutter::EncodableValue("imgsz")] = value;
        }
      }
    } else if (key == "stride") {
      try {
        map[flutter::EncodableValue("stride")] = std::stoi(value);
      } catch (...) {
        map[flutter::EncodableValue("stride")] = value;
      }
    } else if (key == "names" || key == "labels") {
      // Parse YAML list: {0: person, 1: bicycle, ...} or [person, bicycle, ...]
      flutter::EncodableList labels;
      if (!value.empty() && value[0] == '{') {
        // Dict format: extract values.
        size_t pos = 0;
        while (pos < value.size()) {
          auto colon = value.find(':', pos);
          if (colon == std::string::npos) break;
          auto comma = value.find(',', colon);
          if (comma == std::string::npos) comma = value.size();
          std::string label = value.substr(colon + 1, comma - colon - 1);
          // Trim.
          label.erase(0, label.find_first_not_of(" \t"));
          label.erase(label.find_last_not_of(" \t} ") + 1);
          if (!label.empty()) labels.push_back(label);
          pos = comma + 1;
        }
      }
      map[flutter::EncodableValue("labels")] = labels;
    }
  }

  return map;
}

// ── Inspect ──────────────────────────────────────────────────────────────────

flutter::EncodableMap ModelMetadata::Inspect(const std::string& modelPath) {
  std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open model file: " + modelPath);
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  if (fileSize < 4) {
    throw std::runtime_error("Model file too small: " + modelPath);
  }

  // Read the tail of the file (where ZIP metadata lives).
  size_t readSize = std::min(fileSize, kMaxSearchBytes);
  std::vector<uint8_t> tail(readSize);
  file.seekg(static_cast<std::streamoff>(fileSize - readSize));
  file.read(reinterpret_cast<char*>(tail.data()), readSize);
  file.close();

  // Find ZIP end-of-central-directory.
  int64_t eocdOffset = FindZipSignature(tail, readSize);
  if (eocdOffset < 0) {
    // No ZIP metadata — return empty.
    flutter::EncodableMap empty;
    empty[flutter::EncodableValue("path")] = modelPath;
    empty[flutter::EncodableValue("task")] = std::string("");
    empty[flutter::EncodableValue("labels")] = flutter::EncodableList{};
    return empty;
  }

  // Parse EOCD to find central directory offset.
  const uint8_t* eocd = tail.data() + eocdOffset;
  uint32_t cdSize = ReadU32LE(eocd + 12);
  uint32_t cdOffset = ReadU32LE(eocd + 16);

  // The ZIP is appended to the TFLite file, so actual file offsets need
  // adjustment:  fileOffset = modelBase + zipStartInTail + cdOffset.
  size_t zipStartInFile = fileSize - readSize + static_cast<size_t>(eocdOffset);
  // cdOffset is relative to start of the ZIP archive.
  // But Ultralytics appends ZIP at the end of the TFLite file, so
  // the ZIP start = fileSize - (size from ZIP start to EOF).
  // We need to find where the ZIP actually starts.
  // The simplest approach: look for the first central directory header
  // by scanning from cdOffset relative to the ZIP start.

  // For simplicity, search for the first file entry in the ZIP.
  // Look for "metadata.yaml" or similar.
  std::string metadataContent;

  // Scan through central directory entries.
  size_t cdStart = zipStartInFile;
  // Actually, the ZIP starts somewhere before eocdOffset.
  // Let's use a simpler approach: scan the tail for YAML content.
  // Ultralytics metadata is typically a YAML file named "metadata.yaml"
  // stored as the first entry in the ZIP.

  // Search for "task:" pattern in the tail (YAML metadata).
  std::string tailStr(tail.begin(), tail.end());
  auto taskPos = tailStr.find("\ntask:");
  if (taskPos == std::string::npos) taskPos = tailStr.find("task:");

  if (taskPos != std::string::npos) {
    // Extract YAML block — look for the start of the file entry.
    // Find the beginning of the YAML content (go back to find a reasonable start).
    size_t yamlStart = tailStr.rfind('\n', taskPos);
    if (yamlStart == std::string::npos) yamlStart = taskPos;
    else yamlStart += 1;

    // Find the end — look for a ZIP signature or end of meaningful content.
    size_t yamlEnd = tailStr.size();
    for (size_t i = taskPos + 5; i < tailStr.size() - 3; ++i) {
      if (ReadU32LE(tail.data() + i) == kZipCdhSig ||
          ReadU32LE(tail.data() + i) == kZipEndSig) {
        yamlEnd = i;
        break;
      }
    }

    metadataContent = tailStr.substr(yamlStart, yamlEnd - yamlStart);
    return ParseYamlMetadata(metadataContent);
  }

  // No metadata found.
  flutter::EncodableMap empty;
  empty[flutter::EncodableValue("path")] = modelPath;
  empty[flutter::EncodableValue("task")] = std::string("");
  empty[flutter::EncodableValue("labels")] = flutter::EncodableList{};
  return empty;
}

}  // namespace ultralytics_yolo
