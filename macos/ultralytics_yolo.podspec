Pod::Spec.new do |s|
  s.name             = 'ultralytics_yolo'
  s.version          = '0.6.0'
  s.summary          = 'Ultralytics YOLO plugin for Flutter desktop platforms.'
  s.description      = 'Flutter plugin for Ultralytics YOLO computer vision models on Windows, Linux, macOS, and Web.'
  s.homepage         = 'https://github.com/Maicarons/yolo-flutter-app-desktop'
  s.license          = { :type => 'AGPL-3.0', :file => '../LICENSE' }
  s.author           = { 'Ultralytics' => 'hello@ultralytics.com' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'FlutterMacOS'
  s.platform         = :osx, '10.15'
  s.swift_version    = '5.0'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
end
