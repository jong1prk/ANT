cmake_minimum_required(VERSION 2.8)

add_custom_target(
  nil-communication ALL
  COMMAND node-gyp configure
  COMMAND node-gyp build
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/build/Release/communication-api.node ${CMAKE_SOURCE_DIR}/out/bin/api
  COMMENT "Building NIL for Communication API"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
