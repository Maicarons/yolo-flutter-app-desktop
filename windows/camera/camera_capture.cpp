// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "camera_capture.h"

#include <mfapi.h>
#include <mferror.h>
#include <mfobjects.h>
#include <shlwapi.h>

#include <iostream>

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "shlwapi.lib")

namespace ultralytics_yolo {

using Microsoft::WRL::ComPtr;

// ── Lifecycle ────────────────────────────────────────────────────────────────

CameraCapture::CameraCapture() {
  static bool mfInitialized = false;
  if (!mfInitialized) {
    MFStartup(MF_VERSION);
    mfInitialized = true;
  }
}

CameraCapture::~CameraCapture() {
  Stop();
}

// ── EnumerateDevices ─────────────────────────────────────────────────────────

std::vector<CameraDeviceInfo> CameraCapture::EnumerateDevices() {
  std::vector<CameraDeviceInfo> devices;

  ComPtr<IMFAttributes> attrs;
  MFCreateAttributes(&attrs, 1);
  attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                 MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);

  IMFActivate** activates = nullptr;
  UINT32 count = 0;
  if (FAILED(MFEnumDeviceSources(attrs.Get(), &activates, &count))) {
    return devices;
  }

  for (UINT32 i = 0; i < count; ++i) {
    CameraDeviceInfo info;

    // Get symbolic link (unique ID).
    WCHAR symlink[512] = {};
    UINT32 symlinkLen = 0;
    activates[i]->GetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
                            symlink, 512, &symlinkLen);
    if (symlinkLen > 0) {
      info.id = symlink;
    }

    // Get friendly name.
    WCHAR name[256] = {};
    UINT32 nameLen = 0;
    activates[i]->GetString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                            name, 256, &nameLen);
    if (nameLen > 0) {
      info.name = name;
    }

    devices.push_back(info);
    activates[i]->Release();
  }
  CoTaskMemFree(activates);
  return devices;
}

// ── Start ────────────────────────────────────────────────────────────────────

void CameraCapture::Start(int deviceIndex, const std::string& resolution,
                          FrameCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (capturing_) {
    Stop();
  }

  callback_ = std::move(callback);

  // Enumerate and pick device.
  ComPtr<IMFAttributes> devAttrs;
  MFCreateAttributes(&devAttrs, 1);
  devAttrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);

  IMFActivate** activates = nullptr;
  UINT32 count = 0;
  if (FAILED(MFEnumDeviceSources(devAttrs.Get(), &activates, &count)) ||
      count == 0) {
    throw std::runtime_error("No camera devices found");
  }

  if (deviceIndex < 0 || deviceIndex >= static_cast<int>(count)) {
    deviceIndex = 0;
  }

  // Create media source from the selected device.
  HRESULT hr = activates[deviceIndex]->ActivateObject(
      IID_PPV_ARGS(&source_));
  for (UINT32 i = 0; i < count; ++i) activates[i]->Release();
  CoTaskMemFree(activates);

  if (FAILED(hr)) {
    throw std::runtime_error("Failed to activate camera device");
  }

  // Create source reader.
  ComPtr<IMFAttributes> readerAttrs;
  MFCreateAttributes(&readerAttrs, 2);
  readerAttrs->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
  readerAttrs->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);

  hr = MFCreateSourceReaderFromMediaSource(source_.Get(), readerAttrs.Get(),
                                           &reader_);
  if (FAILED(hr)) {
    Cleanup();
    throw std::runtime_error("Failed to create source reader");
  }

  // Set media type — request BGRA format for easy processing.
  ComPtr<IMFMediaType> mediaType;
  MFCreateMediaType(&mediaType);
  mediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
  mediaType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);

  // Set resolution.
  UINT32 targetW = 1280, targetH = 720;
  if (resolution == "1080p") {
    targetW = 1920;
    targetH = 1080;
  } else if (resolution == "480p") {
    targetW = 640;
    targetH = 480;
  }

  MFSetAttributeSize(mediaType.Get(), MF_MT_FRAME_SIZE, targetW, targetH);
  reader_->SetCurrentMediaType(
      static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM), nullptr,
      mediaType.Get());

  // Start the read thread.
  stopRequested_ = false;
  capturing_ = true;
  readThread_ = CreateThread(nullptr, 0, ReadThreadProc, this, 0, nullptr);
}

// ── Stop ─────────────────────────────────────────────────────────────────────

void CameraCapture::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!capturing_) return;
    stopRequested_ = true;
  }

  if (readThread_) {
    WaitForSingleObject(readThread_, 3000);
    CloseHandle(readThread_);
    readThread_ = nullptr;
  }

  Cleanup();
  capturing_ = false;
}

void CameraCapture::Cleanup() {
  reader_.Reset();
  if (source_) {
    source_->Shutdown();
    source_.Reset();
  }
}

// ── Read thread ──────────────────────────────────────────────────────────────

DWORD WINAPI CameraCapture::ReadThreadProc(LPVOID param) {
  static_cast<CameraCapture*>(param)->ReadLoop();
  return 0;
}

void CameraCapture::ReadLoop() {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopRequested_) break;
    }

    DWORD streamIndex = 0;
    DWORD flags = 0;
    LONGLONG timestamp = 0;
    ComPtr<IMFSample> sample;

    HRESULT hr = reader_->ReadSample(
        static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM), 0,
        &streamIndex, &flags, &timestamp, &sample);

    if (FAILED(hr) || !sample) {
      Sleep(10);
      continue;
    }

    if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
      break;
    }

    // Get buffer.
    ComPtr<IMFMediaBuffer> buffer;
    sample->ConvertToContiguousBuffer(&buffer);

    BYTE* data = nullptr;
    DWORD dataLen = 0;
    buffer->Lock(&data, nullptr, &dataLen);

    // Get dimensions from current media type.
    ComPtr<IMFMediaType> mediaType;
    reader_->GetCurrentMediaType(
        static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM),
        &mediaType);

    UINT32 w = 0, h = 0;
    MFGetAttributeSize(mediaType.Get(), MF_MT_FRAME_SIZE, &w, &h);

    if (callback_ && data && w > 0 && h > 0) {
      callback_(data, static_cast<int>(w), static_cast<int>(h));
    }

    buffer->Unlock();
  }
}

}  // namespace ultralytics_yolo
