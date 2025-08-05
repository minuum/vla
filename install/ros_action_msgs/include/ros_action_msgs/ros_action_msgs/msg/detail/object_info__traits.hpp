// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ros_action_msgs:msg/ObjectInfo.idl
// generated code does not contain a copyright notice

#ifndef ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__TRAITS_HPP_
#define ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ros_action_msgs/msg/detail/object_info__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace ros_action_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ObjectInfo & msg,
  std::ostream & out)
{
  out << "{";
  // member: object_id
  {
    out << "object_id: ";
    rosidl_generator_traits::value_to_yaml(msg.object_id, out);
    out << ", ";
  }

  // member: distance
  {
    out << "distance: ";
    rosidl_generator_traits::value_to_yaml(msg.distance, out);
    out << ", ";
  }

  // member: angle
  {
    out << "angle: ";
    rosidl_generator_traits::value_to_yaml(msg.angle, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ObjectInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: object_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "object_id: ";
    rosidl_generator_traits::value_to_yaml(msg.object_id, out);
    out << "\n";
  }

  // member: distance
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "distance: ";
    rosidl_generator_traits::value_to_yaml(msg.distance, out);
    out << "\n";
  }

  // member: angle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "angle: ";
    rosidl_generator_traits::value_to_yaml(msg.angle, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ObjectInfo & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace ros_action_msgs

namespace rosidl_generator_traits
{

[[deprecated("use ros_action_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const ros_action_msgs::msg::ObjectInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  ros_action_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ros_action_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ros_action_msgs::msg::ObjectInfo & msg)
{
  return ros_action_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ros_action_msgs::msg::ObjectInfo>()
{
  return "ros_action_msgs::msg::ObjectInfo";
}

template<>
inline const char * name<ros_action_msgs::msg::ObjectInfo>()
{
  return "ros_action_msgs/msg/ObjectInfo";
}

template<>
struct has_fixed_size<ros_action_msgs::msg::ObjectInfo>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<ros_action_msgs::msg::ObjectInfo>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<ros_action_msgs::msg::ObjectInfo>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // ROS_ACTION_MSGS__MSG__DETAIL__OBJECT_INFO__TRAITS_HPP_
