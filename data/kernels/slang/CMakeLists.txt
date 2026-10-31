#
# compile the Slang kernels to SPIR-V, and to a Metal library on macOS with Xcode
#
# Nothing in darktable loads these yet: the OpenCL code path keeps using the
# .cl sources one directory up.
#

# modules are compiled as part of every kernel that imports them
set(DT_SLANG_MODULES common colorspace hdr_transfer color_conversion rgb_norms noise_generator capture_common)

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

# keep unused parameters, so that src/common/slang_gpu can bind arguments by
# position, as OpenCL does. Only for SPIR-V: slangc 2026.12 crashes on
# colorspaces.slang with -target metal -preserve-params, and 2026.20 on every
# kernel. The Metal backend relies on the kernels using all their parameters.
set(DT_SLANG_SPIRV_FLAGS -preserve-params)

# DT_SLANG_METALLIB comes from the probe for Apple's metal compiler in the top CMakeLists.txt

set(DT_SLANG_OUTPUTS "")
set(DT_SLANG_INSTALL "")

foreach(KERNEL IN ITEMS ${DT_SLANG_KERNELS})
  get_filename_component(KERNAME ${KERNEL} NAME_WE)

  set(SPV "${CMAKE_CURRENT_BINARY_DIR}/${KERNAME}.spv")
  add_custom_command(
    OUTPUT  ${SPV}
    COMMAND ${SLANGC_EXECUTABLE} ${KERNEL} -target spirv ${DT_SLANG_FLAGS} ${DT_SLANG_SPIRV_FLAGS} -o ${SPV}
    DEPENDS ${KERNEL} ${DT_SLANG_MODULE_FILES}
    COMMENT "Compiling Slang kernel ${KERNAME} to SPIR-V"
  )
  list(APPEND DT_SLANG_OUTPUTS ${SPV})
  list(APPEND DT_SLANG_INSTALL ${SPV})

  if(DT_SLANG_METALLIB)
    # one library per entry point: Slang numbers Metal buffer and texture
    # indices across all entry points of a file, and a kernel can only use
    # buffers 0-30. Compiled alone, each entry point starts at 0 again. The
    # libraries can't be linked into one, since every one of them defines
    # Slang's helper functions. The entry points come from the Metal source,
    # since some are declared through macros.
    set(METAL_ALL "${CMAKE_CURRENT_BINARY_DIR}/${KERNAME}.entries.metal")
    execute_process(COMMAND ${SLANGC_EXECUTABLE} ${KERNEL} -target metal ${DT_SLANG_FLAGS} -o ${METAL_ALL}
                    RESULT_VARIABLE SLANGC_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(NOT SLANGC_RESULT EQUAL 0)
      message(FATAL_ERROR "slangc can't translate ${KERNEL} to Metal")
    endif()
    file(STRINGS ${METAL_ALL} ENTRIES REGEX "^\\[\\[kernel\\]\\] void [A-Za-z0-9_]+\\(")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${KERNEL} ${DT_SLANG_MODULE_FILES})
    foreach(ENTRY IN ITEMS ${ENTRIES})
      string(REGEX REPLACE "^\\[\\[kernel\\]\\] void ([A-Za-z0-9_]+)\\(.*" "\\1" ENTRY "${ENTRY}")
      set(METALLIB "${CMAKE_CURRENT_BINARY_DIR}/${KERNAME}.${ENTRY}.metallib")
      # Metal compiles with fast math unless told otherwise, which lets it
      # rearrange the exact additions of Kahan_sum() in common.slang and
      # loosens results that the OpenCL kernels compute without it
      add_custom_command(
        OUTPUT  ${METALLIB}
        COMMAND ${SLANGC_EXECUTABLE} ${KERNEL} -entry ${ENTRY} -stage compute -target metallib ${DT_SLANG_FLAGS} -Xmetal -fno-fast-math -o ${METALLIB}
        DEPENDS ${KERNEL} ${DT_SLANG_MODULE_FILES}
        COMMENT "Compiling Slang kernel ${KERNAME}:${ENTRY} to a Metal library"
      )
      list(APPEND DT_SLANG_OUTPUTS ${METALLIB})
      list(APPEND DT_SLANG_INSTALL ${METALLIB})
    endforeach()
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
