// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ros_action_msgs:msg/ObjectInfo.idl
// generated code does not contain a copyright notice

#ifndef ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__BUILDER_HPP_
#define ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ros_action_msgs/msg/detail/object_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ros_action_msgs
{

namespace msg
{

namespace builder
{

class Init_ObjectInfo_angle
{
public:
  explicit Init_ObjectInfo_angle(::ros_action_msgs::msg::ObjectInfo & msg)
  : msg_(msg)
  {}
  ::ros_action_msgs::msg::ObjectInfo angle(::ros_action_msgs::msg::ObjectInfo::_angle_type arg)
  {
    msg_.angle = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ros_action_msgs::msg::ObjectInfo msg_;
};

class Init_ObjectInfo_distance
{
public:
  explicit Init_ObjectInfo_distance(::ros_action_msgs::msg::ObjectInfo & msg)
  : msg_(msg)
  {}
  Init_ObjectInfo_angle distance(::ros_action_msgs::msg::ObjectInfo::_distance_type arg)
  {
    msg_.distance = std::move(arg);
    return Init_ObjectInfo_angle(msg_);
  }

private:
  ::ros_action_msgs::msg::ObjectInfo msg_;
};

class Init_ObjectInfo_object_id
{
public:
  Init_ObjectInfo_object_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ObjectInfo_distance object_id(::ros_action_msgs::msg::ObjectInfo::_object_id_type arg)
  {
    msg_.object_id = std::move(arg);
    return Init_ObjectInfo_distance(msg_);
  }

private:
  ::ros_action_msgs::msg::ObjectInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ros_action_msgs::msg::ObjectInfo>()
{
  return ros_action_msgs::msg::builder::Init_ObjectInfo_object_id();
}

}  // namespace ros_action_msgs

#endif  // ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__BUILDER_HPP_
