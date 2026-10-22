#
# compile the Slang kernels to SPIR-V, and to a Metal library on macOS with Xcode
#
# Nothing in darktable loads these yet: the OpenCL code path keeps using the
# .cl sources one directory up.
#

# modules are compiled as part of every kernel that imports them
set(DT_SLANG_MODULES common colorspace hdr_transfer color_conversion rgb_norms)

set(DT_SLANG_MODULE_FILES "")
foreach(MODULE IN ITEMS ${DT_SLANG_MODULES})
  list(APPEND DT_SLANG_MODULE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}.slang")
endforeach()

file(GLOB DT_SLANG_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.slang")
set(DT_SLANG_KERNELS ${DT_SLANG_SOURCES})
list(REMOVE_ITEM DT_SLANG_KERNELS ${DT_SLANG_MODULE_FILES})

# 31106/31107: an ImageIn/ImageOut argument holds a texture next to the uniform
# scalars, so Slang moves the texture to its own binding, which is what we want
set(DT_SLANG_FLAGS -I ${CMAKE_CURRENT_SOURCE_DIR} -fvk-use-entrypoint-name -warnings-disable 31106,31107)

# slangc hands .metallib builds to Apple's metal compiler, which ships with
# Xcode but not with the Command Line Tools
set(DT_SLANG_METALLIB OFF)
if(APPLE)
  execute_process(COMMAND xcrun -sdk macosx -f metal
                  RESULT_VARIABLE XCRUN_METAL_RESULT OUTPUT_QUIET ERROR_QUIET)
  if(XCRUN_METAL_RESULT EQUAL 0)
    set(DT_SLANG_METALLIB ON)
  else()
    message(STATUS "Metal compiler not found (needs Xcode), Slang kernels are translated to Metal source only")
  endif()
endif()

set(DT_SLANG_OUTPUTS "")
set(DT_SLANG_INSTALL "")

foreach(KERNEL IN ITEMS ${DT_SLANG_KERNELS})
  get_filename_component(KERNAME ${KERNEL} NAME_WE)

  set(SPV "${CMAKE_CURRENT_BINARY_DIR}/${KERNAME}.spv")
  add_custom_command(
    OUTPUT  ${SPV}
    COMMAND ${SLANGC_EXECUTABLE} ${KERNEL} -target spirv ${DT_SLANG_FLAGS} -o ${SPV}
    DEPENDS ${KERNEL} ${DT_SLANG_MODULE_FILES}
    COMMENT "Compiling Slang kernel ${KERNAME} to SPIR-V"
  )
  list(APPEND DT_SLANG_OUTPUTS ${SPV})
  list(APPEND DT_SLANG_INSTALL ${SPV})

  if(DT_SLANG_METALLIB)
    set(METALLIB "${CMAKE_CURRENT_BINARY_DIR}/${KERNAME}.metallib")
    add_custom_command(
      OUTPUT  ${METALLIB}
      COMMAND ${SLANGC_EXECUTABLE} ${KERNEL} -target metallib ${DT_SLANG_FLAGS} -o ${METALLIB}
      DEPENDS ${KERNEL} ${DT_SLANG_MODULE_FILES}
      COMMENT "Compiling Slang kernel ${KERNAME} to a Metal library"
    )
    list(APPEND DT_SLANG_OUTPUTS ${METALLIB})
    list(APPEND DT_SLANG_INSTALL ${METALLIB})
  else()
    # without Apple's metal compiler, generating the source still catches
    # kernels that do not translate to Metal, and it can be compiled at runtime
    set(METAL "${CMAKE_CURRENT_BINARY_DIR}/${KERNAME}.metal")
    add_custom_command(
      OUTPUT  ${METAL}
      COMMAND ${SLANGC_EXECUTABLE} ${KERNEL} -target metal ${DT_SLANG_FLAGS} -o ${METAL}
      DEPENDS ${KERNEL} ${DT_SLANG_MODULE_FILES}
      COMMENT "Translating Slang kernel ${KERNAME} to Metal source"
    )
    list(APPEND DT_SLANG_OUTPUTS ${METAL})
  endif()
endforeach()

add_custom_target(slang_kernels ALL DEPENDS ${DT_SLANG_OUTPUTS})

install(FILES ${DT_SLANG_INSTALL} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/darktable/kernels/slang COMPONENT DTApplication)
