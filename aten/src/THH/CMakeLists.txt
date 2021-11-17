set(ATen_HIP_INCLUDE ${ATen_HIP_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

configure_file(THHGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THHGeneral.h")

set(ATen_HIP_SRCS ${ATen_HIP_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THHCachingHostAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THHGeneral.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THHStorageCopy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensor.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/THHReduceApplyUtils.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHSleep.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHStorage.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHStorageCopy.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensor.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorCopy.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorMath.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorMathMagma.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorMathPairwise.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorMathReduce.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorScatterGather.hip
  PARENT_SCOPE)

install(FILES
          THH.h
          ${CMAKE_CURRENT_BINARY_DIR}/THHGeneral.h
          THHGeneral.hpp
          THHSleep.h
          THHStorage.h
          THHStorageCopy.h
          THHTensor.h
          THHTensorCopy.h
          THHTensorCopy.hpp
          THHTensorMath.h
          THHApply.cuh
          THHReduceApplyUtils.cuh
          THHTensorMathReduce.cuh
          THHAsmUtils.cuh
          THHAtomics.cuh
          THHScanUtils.cuh
          THHAllocator.h
          THHCachingHostAllocator.h
          THHDeviceUtils.cuh
          THHDeviceTensor.cuh
          THHDeviceTensor-inl.cuh
          THHDeviceTensorUtils.cuh
          THHDeviceTensorUtils-inl.cuh
          THHGenerateAllTypes.h
          THHGenerateBFloat16Type.h
          THHGenerateBoolType.h
          THHGenerateByteType.h
          THHGenerateCharType.h
          THHGenerateShortType.h
          THHGenerateIntType.h
          THHGenerateLongType.h
          THHGenerateHalfType.h
          THHGenerateFloatType.h
          THHGenerateFloatTypes.h
          THHGenerateDoubleType.h
          THHGenerateComplexFloatType.h
          THHGenerateComplexTypes.h
          THHGenerateComplexDoubleType.h
          THHIntegerDivider.cuh
          THHNumerics.cuh
          THHTensorInfo.cuh
          THHTensorTypeUtils.cuh
          THHTensorMathMagma.h
          THHThrustAllocator.cuh
          # See Note [TH abstraction violation]
          THHTensor.hpp
          THHStorage.hpp
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THH")

install(FILES
          generic/THHStorage.cpp
          generic/THHStorage.hip
          generic/THHStorage.h
          generic/THHTensor.cpp
          generic/THHTensor.hip
          generic/THHTensor.h
          generic/THHTensor.hpp
          generic/THHStorageCopy.cpp
          generic/THHStorageCopy.hip
          generic/THHStorageCopy.h
          generic/THHTensorCopy.hip
          generic/THHTensorCopy.h
          generic/THHTensorMath.h
          generic/THHTensorMath.hip
          generic/THHTensorMathMagma.h
          generic/THHTensorMathMagma.cpp
          generic/THHTensorMathPairwise.h
          generic/THHTensorMathPairwise.hip
          generic/THHTensorScatterGather.h
          generic/THHTensorScatterGather.hip
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THH/generic")
