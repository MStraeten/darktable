# compares the Slang kernels, run through Vulkan, with the OpenCL kernels they
# were ported from. Mesa's lavapipe and PoCL are enough, no GPU is needed.

find_package(Vulkan QUIET)
find_package(OpenCL QUIET)

if(NOT Vulkan_FOUND OR NOT OpenCL_FOUND)
  message(STATUS "Slang kernel parity test disabled: needs the Vulkan and OpenCL loaders")
  return()
endif()

add_executable(darktable-test-slang-parity parity.c)
target_link_libraries(darktable-test-slang-parity Vulkan::Vulkan OpenCL::OpenCL m)
add_dependencies(darktable-test-slang-parity slang_kernels)

add_test(NAME slang_kernel_parity
         COMMAND darktable-test-slang-parity ${CMAKE_SOURCE_DIR}/data/kernels ${DARKTABLE_DATADIR}/kernels/slang)
