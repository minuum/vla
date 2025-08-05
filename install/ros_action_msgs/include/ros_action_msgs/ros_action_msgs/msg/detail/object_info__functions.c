// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ros_action_msgs:msg/ObjectInfo.idl
// generated code does not contain a copyright notice
#include "ros_action_msgs/msg/detail/object_info__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `object_id`
#include "rosidl_runtime_c/string_functions.h"

bool
ros_action_msgs__msg__ObjectInfo__init(ros_action_msgs__msg__ObjectInfo * msg)
{
  if (!msg) {
    return false;
  }
  // object_id
  if (!rosidl_runtime_c__String__init(&msg->object_id)) {
    ros_action_msgs__msg__ObjectInfo__fini(msg);
    return false;
  }
  // distance
  // angle
  return true;
}

void
ros_action_msgs__msg__ObjectInfo__fini(ros_action_msgs__msg__ObjectInfo * msg)
{
  if (!msg) {
    return;
  }
  // object_id
  rosidl_runtime_c__String__fini(&msg->object_id);
  // distance
  // angle
}

bool
ros_action_msgs__msg__ObjectInfo__are_equal(const ros_action_msgs__msg__ObjectInfo * lhs, const ros_action_msgs__msg__ObjectInfo * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // object_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->object_id), &(rhs->object_id)))
  {
    return false;
  }
  // distance
  if (lhs->distance != rhs->distance) {
    return false;
  }
  // angle
  if (lhs->angle != rhs->angle) {
    return false;
  }
  return true;
}

bool
ros_action_msgs__msg__ObjectInfo__copy(
  const ros_action_msgs__msg__ObjectInfo * input,
  ros_action_msgs__msg__ObjectInfo * output)
{
  if (!input || !output) {
    return false;
  }
  // object_id
  if (!rosidl_runtime_c__String__copy(
      &(input->object_id), &(output->object_id)))
  {
    return false;
  }
  // distance
  output->distance = input->distance;
  // angle
  output->angle = input->angle;
  return true;
}

ros_action_msgs__msg__ObjectInfo *
ros_action_msgs__msg__ObjectInfo__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ros_action_msgs__msg__ObjectInfo * msg = (ros_action_msgs__msg__ObjectInfo *)allocator.allocate(sizeof(ros_action_msgs__msg__ObjectInfo), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ros_action_msgs__msg__ObjectInfo));
  bool success = ros_action_msgs__msg__ObjectInfo__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ros_action_msgs__msg__ObjectInfo__destroy(ros_action_msgs__msg__ObjectInfo * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ros_action_msgs__msg__ObjectInfo__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ros_action_msgs__msg__ObjectInfo__Sequence__init(ros_action_msgs__msg__ObjectInfo__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ros_action_msgs__msg__ObjectInfo * data = NULL;

  if (size) {
    data = (ros_action_msgs__msg__ObjectInfo *)allocator.zero_allocate(size, sizeof(ros_action_msgs__msg__ObjectInfo), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ros_action_msgs__msg__ObjectInfo__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ros_action_msgs__msg__ObjectInfo__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
ros_action_msgs__msg__ObjectInfo__Sequence__fini(ros_action_msgs__msg__ObjectInfo__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      ros_action_msgs__msg__ObjectInfo__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

ros_action_msgs__msg__ObjectInfo__Sequence *
ros_action_msgs__msg__ObjectInfo__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ros_action_msgs__msg__ObjectInfo__Sequence * array = (ros_action_msgs__msg__ObjectInfo__Sequence *)allocator.allocate(sizeof(ros_action_msgs__msg__ObjectInfo__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ros_action_msgs__msg__ObjectInfo__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ros_action_msgs__msg__ObjectInfo__Sequence__destroy(ros_action_msgs__msg__ObjectInfo__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ros_action_msgs__msg__ObjectInfo__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ros_action_msgs__msg__ObjectInfo__Sequence__are_equal(const ros_action_msgs__msg__ObjectInfo__Sequence * lhs, const ros_action_msgs__msg__ObjectInfo__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ros_action_msgs__msg__ObjectInfo__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ros_action_msgs__msg__ObjectInfo__Sequence__copy(
  const ros_action_msgs__msg__ObjectInfo__Sequence * input,
  ros_action_msgs__msg__ObjectInfo__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ros_action_msgs__msg__ObjectInfo);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ros_action_msgs__msg__ObjectInfo * data =
      (ros_action_msgs__msg__ObjectInfo *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ros_action_msgs__msg__ObjectInfo__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ros_action_msgs__msg__ObjectInfo__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ros_action_msgs__msg__ObjectInfo__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
