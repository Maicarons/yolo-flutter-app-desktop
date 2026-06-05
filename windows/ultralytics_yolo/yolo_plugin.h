// Minimal header for Flutter plugin registration.
// The full implementation lives in the plugin DLL; this header only exposes
// the C entry point so the runner can call it without pulling in ONNX Runtime.

#ifndef FLUTTER_PLUGIN_ULTRALYTICS_YOLO_PLUGIN_REGISTRAR_H_
#define FLUTTER_PLUGIN_ULTRALYTICS_YOLO_PLUGIN_REGISTRAR_H_

#include <flutter_plugin_registrar.h>

// C entry point exported by the plugin DLL.
#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllimport) void YoloPluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar);

#ifdef __cplusplus
}
#endif

#endif  // FLUTTER_PLUGIN_ULTRALYTICS_YOLO_PLUGIN_REGISTRAR_H_
