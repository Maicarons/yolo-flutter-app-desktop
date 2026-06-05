cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME "ultralytics_yolo")
project(${PROJECT_NAME} LANGUAGES CXX)

set(PLUGIN_NAME "ultralytics_yolo_plugin")

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
# Uses a pre-downloaded ONNX Runtime SDK for Linux.
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/onnxruntime-linux-x64-1.20.1")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include")
  message(FATAL_ERROR
    "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}. "
    "Download from https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1 "
    "and extract to linux/third_party/onnxruntime/, or set ONNXRUNTIME_ROOT.")
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
)

target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

# Copy onnxruntime.so into the binary output directory.
file(GLOB ORT_LIBS "${ORT_LIB_DIR}/*.so*")
foreach(ORT_LIB ${ORT_LIBS})
  add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_LIB}" "$<TARGET_FILE_DIR:${PLUGIN_NAME}>"
  )
endforeach()

# Tell Flutter to bundle these libraries.
set(${PLUGIN_NAME}_bundled_libraries ${ORT_LIBS} PARENT_SCOPE)
