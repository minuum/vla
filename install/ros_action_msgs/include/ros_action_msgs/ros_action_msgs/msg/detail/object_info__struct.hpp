// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ros_action_msgs:msg/ObjectInfo.idl
// generated code does not contain a copyright notice

#ifndef ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__STRUCT_HPP_
#define ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ros_action_msgs__msg__ObjectInfo __attribute__((deprecated))
#else
# define DEPRECATED__ros_action_msgs__msg__ObjectInfo __declspec(deprecated)
#endif

namespace ros_action_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ObjectInfo_
{
  using Type = ObjectInfo_<ContainerAllocator>;

  explicit ObjectInfo_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->object_id = "";
      this->distance = 0.0;
      this->angle = 0.0;
    }
  }

  explicit ObjectInfo_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : object_id(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->object_id = "";
      this->distance = 0.0;
      this->angle = 0.0;
    }
  }

  // field types and members
  using _object_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _object_id_type object_id;
  using _distance_type =
    double;
  _distance_type distance;
  using _angle_type =
    double;
  _angle_type angle;

  // setters for named parameter idiom
  Type & set__object_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->object_id = _arg;
    return *this;
  }
  Type & set__distance(
    const double & _arg)
  {
    this->distance = _arg;
    return *this;
  }
  Type & set__angle(
    const double & _arg)
  {
    this->angle = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ros_action_msgs::msg::ObjectInfo_<ContainerAllocator> *;
  using ConstRawPtr =
    const ros_action_msgs::msg::ObjectInfo_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ros_action_msgs::msg::ObjectInfo_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ros_action_msgs::msg::ObjectInfo_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ros_action_msgs::msg::ObjectInfo_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ros_action_msgs::msg::ObjectInfo_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ros_action_msgs::msg::ObjectInfo_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ros_action_msgs::msg::ObjectInfo_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ros_action_msgs::msg::ObjectInfo_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ros_action_msgs::msg::ObjectInfo_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ros_action_msgs__msg__ObjectInfo
    std::shared_ptr<ros_action_msgs::msg::ObjectInfo_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ros_action_msgs__msg__ObjectInfo
    std::shared_ptr<ros_action_msgs::msg::ObjectInfo_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ObjectInfo_ & other) const
  {
    if (this->object_id != other.object_id) {
      return false;
    }
    if (this->distance != other.distance) {
      return false;
    }
    if (this->angle != other.angle) {
      return false;
    }
    return true;
  }
  bool operator!=(const ObjectInfo_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ObjectInfo_

// alias to use template instance with default allocator
using ObjectInfo =
  ros_action_msgs::msg::ObjectInfo_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ros_action_msgs

#endif  // ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__STRUCT_HPP_
