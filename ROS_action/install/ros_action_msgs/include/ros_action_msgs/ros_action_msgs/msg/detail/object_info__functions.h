// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from ros_action_msgs:msg/ObjectInfo.idl
// generated code does not contain a copyright notice

#ifndef ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__FUNCTIONS_H_
#define ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "ros_action_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "ros_action_msgs/msg/detail/object_info__struct.h"

/// Initialize msg/ObjectInfo message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * ros_action_msgs__msg__ObjectInfo
 * )) before or use
 * ros_action_msgs__msg__ObjectInfo__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
bool
ros_action_msgs__msg__ObjectInfo__init(ros_action_msgs__msg__ObjectInfo * msg);

/// Finalize msg/ObjectInfo message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
void
ros_action_msgs__msg__ObjectInfo__fini(ros_action_msgs__msg__ObjectInfo * msg);

/// Create msg/ObjectInfo message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * ros_action_msgs__msg__ObjectInfo__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
ros_action_msgs__msg__ObjectInfo *
ros_action_msgs__msg__ObjectInfo__create();

/// Destroy msg/ObjectInfo message.
/**
 * It calls
 * ros_action_msgs__msg__ObjectInfo__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
void
ros_action_msgs__msg__ObjectInfo__destroy(ros_action_msgs__msg__ObjectInfo * msg);

/// Check for msg/ObjectInfo message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
bool
ros_action_msgs__msg__ObjectInfo__are_equal(const ros_action_msgs__msg__ObjectInfo * lhs, const ros_action_msgs__msg__ObjectInfo * rhs);

/// Copy a msg/ObjectInfo message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
bool
ros_action_msgs__msg__ObjectInfo__copy(
  const ros_action_msgs__msg__ObjectInfo * input,
  ros_action_msgs__msg__ObjectInfo * output);

/// Initialize array of msg/ObjectInfo messages.
/**
 * It allocates the memory for the number of elements and calls
 * ros_action_msgs__msg__ObjectInfo__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
bool
ros_action_msgs__msg__ObjectInfo__Sequence__init(ros_action_msgs__msg__ObjectInfo__Sequence * array, size_t size);

/// Finalize array of msg/ObjectInfo messages.
/**
 * It calls
 * ros_action_msgs__msg__ObjectInfo__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
void
ros_action_msgs__msg__ObjectInfo__Sequence__fini(ros_action_msgs__msg__ObjectInfo__Sequence * array);

/// Create array of msg/ObjectInfo messages.
/**
 * It allocates the memory for the array and calls
 * ros_action_msgs__msg__ObjectInfo__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
ros_action_msgs__msg__ObjectInfo__Sequence *
ros_action_msgs__msg__ObjectInfo__Sequence__create(size_t size);

/// Destroy array of msg/ObjectInfo messages.
/**
 * It calls
 * ros_action_msgs__msg__ObjectInfo__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
void
ros_action_msgs__msg__ObjectInfo__Sequence__destroy(ros_action_msgs__msg__ObjectInfo__Sequence * array);

/// Check for msg/ObjectInfo message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
bool
ros_action_msgs__msg__ObjectInfo__Sequence__are_equal(const ros_action_msgs__msg__ObjectInfo__Sequence * lhs, const ros_action_msgs__msg__ObjectInfo__Sequence * rhs);

/// Copy an array of msg/ObjectInfo messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_ros_action_msgs
bool
ros_action_msgs__msg__ObjectInfo__Sequence__copy(
  const ros_action_msgs__msg__ObjectInfo__Sequence * input,
  ros_action_msgs__msg__ObjectInfo__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__FUNCTIONS_H_
