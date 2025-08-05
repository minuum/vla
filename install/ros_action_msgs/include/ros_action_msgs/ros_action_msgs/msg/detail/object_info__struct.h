// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ros_action_msgs:msg/ObjectInfo.idl
// generated code does not contain a copyright notice

#ifndef ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__STRUCT_H_
#define ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'object_id'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/ObjectInfo in the package ros_action_msgs.
/**
  * 객체 식별 정보
 */
typedef struct ros_action_msgs__msg__ObjectInfo
{
  rosidl_runtime_c__String object_id;
  /// 단위: 미터
  double distance;
  /// 단위: 라디안 (정면 기준)
  double angle;
} ros_action_msgs__msg__ObjectInfo;

// Struct for a sequence of ros_action_msgs__msg__ObjectInfo.
typedef struct ros_action_msgs__msg__ObjectInfo__Sequence
{
  ros_action_msgs__msg__ObjectInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ros_action_msgs__msg__ObjectInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__STRUCT_H_
