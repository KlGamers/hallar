core = File.expand_path('../../core', File.realpath(__dir__))

Pod::Spec.new do |s|
  s.name             = 'hallar'
  s.version          = '0.1.0'
  s.summary          = 'Búsqueda semántica e híbrida sin internet.'
  s.homepage         = 'https://github.com/KlGamers/hallar'
  s.license          = { :type => 'MIT' }
  s.author           = { 'KlGamers' => 'https://github.com/KlGamers' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*.cpp'
  s.platform         = :osx, '10.15'
  s.dependency 'FlutterMacOS'
  s.library          = 'c++'
  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'HEADER_SEARCH_PATHS' => ["#{core}/include", "#{core}/src", "#{core}/third_party/vecdex/include"].map { |d| "\"#{d}\"" }.join(' '),
    'DEFINES_MODULE' => 'YES',
  }
end
