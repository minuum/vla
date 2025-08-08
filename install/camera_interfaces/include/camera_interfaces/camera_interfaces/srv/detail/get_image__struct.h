// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from camera_interfaces:srv/GetImage.idl
// generated code does not contain a copyright notice

#ifndef CAMERA_INTERFACES__SRV__DETAIL__GET_IMAGE__STRUCT_H_
#define CAMERA_INTERFACES__SRV__DETAIL__GET_IMAGE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetImage in the package camera_interfaces.
typedef struct camera_interfaces__srv__GetImage_Request
{
  uint8_t structure_needs_at_least_one_member;
} camera_interfaces__srv__GetImage_Request;

// Struct for a sequence of camera_interfaces__srv__GetImage_Request.
typedef struct camera_interfaces__srv__GetImage_Request__Sequence
{
  camera_interfaces__srv__GetImage_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} camera_interfaces__srv__GetImage_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'image'
#include "sensor_msgs/msg/detail/image__struct.h"

/// Struct defined in srv/GetImage in the package camera_interfaces.
typedef struct camera_interfaces__srv__GetImage_Response
{
  sensor_msgs__msg__Image image;
} camera_interfaces__srv__GetImage_Response;

// Struct for a sequence of camera_interfaces__srv__GetImage_Response.
typedef struct camera_interfaces__srv__GetImage_Response__Sequence
{
  camera_interfaces__srv__GetImage_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} camera_interfaces__srv__GetImage_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CAMERA_INTERFACES__SRV__DETAIL__GET_IMAGE__STRUCT_H_
