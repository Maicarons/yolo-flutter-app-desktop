// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "camera_capture.h"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace ultralytics_yolo {

CameraCapture::CameraCapture() = default;
CameraCapture::~CameraCapture() { Stop(); }

// ── EnumerateDevices ─────────────────────────────────────────────────────────

std::vector<CameraDeviceInfo> CameraCapture::EnumerateDevices() {
  std::vector<CameraDeviceInfo> devices;
  for (int i = 0; i < 64; ++i) {
    std::string path = "/dev/video" + std::to_string(i);
    if (!std::filesystem::exists(path)) continue;

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) continue;

    struct v4l2_capability cap {};
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0 &&
        (cap.device_caps & V4L2_CAP_VIDEO_CAPTURE)) {
      CameraDeviceInfo info;
      info.id = path;
      info.name = reinterpret_cast<const char*>(cap.card);
      devices.push_back(info);
    }
    close(fd);
  }
  return devices;
}

// ── Start ────────────────────────────────────────────────────────────────────

void CameraCapture::Start(int deviceIndex, const std::string& resolution,
                          FrameCallback callback) {
  Stop();
  callback_ = std::move(callback);

  auto devices = EnumerateDevices();
  if (devices.empty()) throw std::runtime_error("No camera devices found");
  if (deviceIndex < 0 || deviceIndex >= static_cast<int>(devices.size()))
    deviceIndex = 0;

  fd_ = open(devices[deviceIndex].id.c_str(), O_RDWR);
  if (fd_ < 0) throw std::runtime_error("Failed to open camera");

  // Set format.
  struct v4l2_format fmt {};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = 1280;
  fmt.fmt.pix.height = 720;
  if (resolution == "1080p") {
    fmt.fmt.pix.width = 1920;
    fmt.fmt.pix.height = 1080;
  } else if (resolution == "480p") {
    fmt.fmt.pix.width = 640;
    fmt.fmt.pix.height = 480;
  }
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  ioctl(fd_, VIDIOC_S_FMT, &fmt);

  // Request buffers.
  struct v4l2_requestbuffers req {};
  req.count = 1;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  ioctl(fd_, VIDIOC_REQBUFS, &req);

  // Map buffer.
  struct v4l2_buffer buf {};
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.index = 0;
  ioctl(fd_, VIDIOC_QUERYBUF, &buf);

  void* buffer = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_, buf.m.offset);

  // Start stream.
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  ioctl(fd_, VIDIOC_STREAMON, &type);

  capturing_ = true;
  stopRequested_ = false;
  readThread_ = std::thread([this, buf, buffer]() {
    while (!stopRequested_) {
      struct v4l2_buffer readBuf {};
      readBuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      readBuf.memory = V4L2_MEMORY_MMAP;
      readBuf.index = 0;

      ioctl(fd_, VIDIOC_QBUF, &readBuf);
      if (ioctl(fd_, VIDIOC_DQBUF, &readBuf) < 0) {
        usleep(10000);
        continue;
      }

      // Get current format for dimensions.
      struct v4l2_format currentFmt {};
      currentFmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      ioctl(fd_, VIDIOC_G_FMT, &currentFmt);

      if (callback_) {
        callback_(static_cast<const uint8_t*>(buffer),
                  currentFmt.fmt.pix.width, currentFmt.fmt.pix.height);
      }
    }

    munmap(buffer, buf.length);
  });
}

// ── Stop ─────────────────────────────────────────────────────────────────────

void CameraCapture::Stop() {
  if (!capturing_) return;
  stopRequested_ = true;
  if (readThread_.joinable()) readThread_.join();

  if (fd_ >= 0) {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
    close(fd_);
    fd_ = -1;
  }
  capturing_ = false;
}

}  // namespace ultralytics_yolo
