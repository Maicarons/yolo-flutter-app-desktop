cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME "ultralytics_yolo")
project(${PROJECT_NAME} LANGUAGES CXX)

set(PLUGIN_NAME "ultralytics_yolo_plugin")

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
# Uses a pre-downloaded ONNX Runtime SDK.  Set ONNXRUNTIME_ROOT to the extracted
# directory (must contain include/ and lib/).  Falls back to the vendored copy
# under third_party/onnxruntime/.
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/onnxruntime-win-x64-1.20.1")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include")
  message(FATAL_ERROR
    "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}. "
    "Download from https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1 "
    "and extract to windows/third_party/onnxruntime/, or set ONNXRUNTIME_ROOT.")
endif()

set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ORT_LIB_DIR     "${ONNXRUNTIME_ROOT}/lib")

# ── Plugin target ─────────────────────────────────────────────────────────────
add_library(${PLUGIN_NAME} SHARED
  "yolo_plugin.cpp"
  "inference/yolo_engine.cpp"
  "inference/model_metadata.cpp"
  "camera/camera_capture.cpp"
  "camera/camera_texture.cpp"
)

target_include_directories(${PLUGIN_NAME} PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${ORT_INCLUDE_DIR}"
)

target_link_directories(${PLUGIN_NAME} PRIVATE "${ORT_LIB_DIR}")
target_link_libraries(${PLUGIN_NAME} PRIVATE
  flutter
  flutter_wrapper_plugin
  onnxruntime
  onnxruntime_providers_shared
)

target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

# Copy onnxruntime.dll into the binary output directory so it is found at
# runtime when the plugin DLL loads.  Also register them as bundled libraries
# so the Flutter build system copies them next to the runner executable.
file(GLOB ORT_DLLS "${ORT_LIB_DIR}/*.dll")
foreach(ORT_DLL ${ORT_DLLS})
  add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_DLL}" "$<TARGET_FILE_DIR:${PLUGIN_NAME}>"
  )
endforeach()

# Tell Flutter to bundle these DLLs alongside the runner executable.
set(${PLUGIN_NAME}_bundled_libraries ${ORT_DLLS} PARENT_SCOPE)
