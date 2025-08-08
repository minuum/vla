// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from camera_interfaces:srv/GetImage.idl
// generated code does not contain a copyright notice

#ifndef CAMERA_INTERFACES__SRV__DETAIL__GET_IMAGE__TRAITS_HPP_
#define CAMERA_INTERFACES__SRV__DETAIL__GET_IMAGE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "camera_interfaces/srv/detail/get_image__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace camera_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetImage_Request & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetImage_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetImage_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace camera_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use camera_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const camera_interfaces::srv::GetImage_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  camera_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use camera_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const camera_interfaces::srv::GetImage_Request & msg)
{
  return camera_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<camera_interfaces::srv::GetImage_Request>()
{
  return "camera_interfaces::srv::GetImage_Request";
}

template<>
inline const char * name<camera_interfaces::srv::GetImage_Request>()
{
  return "camera_interfaces/srv/GetImage_Request";
}

template<>
struct has_fixed_size<camera_interfaces::srv::GetImage_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<camera_interfaces::srv::GetImage_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<camera_interfaces::srv::GetImage_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'image'
#include "sensor_msgs/msg/detail/image__traits.hpp"

namespace camera_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetImage_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: image
  {
    out << "image: ";
    to_flow_style_yaml(msg.image, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetImage_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: image
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "image:\n";
    to_block_style_yaml(msg.image, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetImage_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace camera_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use camera_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const camera_interfaces::srv::GetImage_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  camera_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use camera_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const camera_interfaces::srv::GetImage_Response & msg)
{
  return camera_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<camera_interfaces::srv::GetImage_Response>()
{
  return "camera_interfaces::srv::GetImage_Response";
}

template<>
inline const char * name<camera_interfaces::srv::GetImage_Response>()
{
  return "camera_interfaces/srv/GetImage_Response";
}

template<>
struct has_fixed_size<camera_interfaces::srv::GetImage_Response>
  : std::integral_constant<bool, has_fixed_size<sensor_msgs::msg::Image>::value> {};

template<>
struct has_bounded_size<camera_interfaces::srv::GetImage_Response>
  : std::integral_constant<bool, has_bounded_size<sensor_msgs::msg::Image>::value> {};

template<>
struct is_message<camera_interfaces::srv::GetImage_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<camera_interfaces::srv::GetImage>()
{
  return "camera_interfaces::srv::GetImage";
}

template<>
inline const char * name<camera_interfaces::srv::GetImage>()
{
  return "camera_interfaces/srv/GetImage";
}

template<>
struct has_fixed_size<camera_interfaces::srv::GetImage>
  : std::integral_constant<
    bool,
    has_fixed_size<camera_interfaces::srv::GetImage_Request>::value &&
    has_fixed_size<camera_interfaces::srv::GetImage_Response>::value
  >
{
};

template<>
struct has_bounded_size<camera_interfaces::srv::GetImage>
  : std::integral_constant<
    bool,
    has_bounded_size<camera_interfaces::srv::GetImage_Request>::value &&
    has_bounded_size<camera_interfaces::srv::GetImage_Response>::value
  >
{
};

template<>
struct is_service<camera_interfaces::srv::GetImage>
  : std::true_type
{
};

template<>
struct is_service_request<camera_interfaces::srv::GetImage_Request>
  : std::true_type
{
};

template<>
struct is_service_response<camera_interfaces::srv::GetImage_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // CAMERA_INTERFACES__SRV__DETAIL__GET_IMAGE__TRAITS_HPP_
