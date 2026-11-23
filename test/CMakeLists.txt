configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NPU_OPT_TEST_DEPENDS
  npu-opt
  FileCheck
)

add_lit_testsuite(check-npu-opt "Running the npu-opt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${NPU_OPT_TEST_DEPENDS}
)
set_target_properties(check-npu-opt PROPERTIES FOLDER "Tests")
