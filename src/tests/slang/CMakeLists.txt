# compares the Slang kernels, run through Vulkan, or Metal on macOS, with the
# OpenCL kernels they were ported from. Mesa's lavapipe and PoCL are enough,
# no GPU is needed. Without a device the test is skipped.

find_package(OpenCL QUIET)

if(NOT OpenCL_FOUND)
  message(STATUS "Slang kernel parity test disabled: needs the OpenCL loader")
  return()
endif()

if(APPLE AND DT_SLANG_METALLIB)
  enable_language(OBJC)
  add_executable(darktable-test-slang-parity parity.c parity_metal.m)
  target_compile_definitions(darktable-test-slang-parity PRIVATE PARITY_METAL)
  # _DARWIN_C_SOURCE, as for slang_gpu_metal.m: the _XOPEN_SOURCE darktable
  # builds with hides the MAC_OS_VERSION_* macros the SDK check needs
  set_source_files_properties(parity_metal.m PROPERTIES COMPILE_OPTIONS "-fobjc-arc;-D_DARWIN_C_SOURCE")
  target_link_libraries(darktable-test-slang-parity OpenCL::OpenCL "-framework Metal" "-framework Foundation" m)
else()
  find_package(Vulkan QUIET)
  if(NOT Vulkan_FOUND)
    message(STATUS "Slang kernel parity test disabled: needs the Vulkan loader")
    return()
  endif()
  add_executable(darktable-test-slang-parity parity.c)
  target_link_libraries(darktable-test-slang-parity Vulkan::Vulkan OpenCL::OpenCL m)
endif()
add_dependencies(darktable-test-slang-parity slang_kernels)

add_test(NAME slang_kernel_parity
         COMMAND darktable-test-slang-parity ${CMAKE_SOURCE_DIR}/data/kernels ${DARKTABLE_DATADIR}/kernels/slang)
set_tests_properties(slang_kernel_parity PROPERTIES SKIP_RETURN_CODE 77)
