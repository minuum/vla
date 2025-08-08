// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from camera_interfaces:srv/GetImage.idl
// generated code does not contain a copyright notice

#ifndef CAMERA_INTERFACES__SRV__DETAIL__GET_IMAGE__BUILDER_HPP_
#define CAMERA_INTERFACES__SRV__DETAIL__GET_IMAGE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "camera_interfaces/srv/detail/get_image__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace camera_interfaces
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::camera_interfaces::srv::GetImage_Request>()
{
  return ::camera_interfaces::srv::GetImage_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace camera_interfaces


namespace camera_interfaces
{

namespace srv
{

namespace builder
{

class Init_GetImage_Response_image
{
public:
  Init_GetImage_Response_image()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::camera_interfaces::srv::GetImage_Response image(::camera_interfaces::srv::GetImage_Response::_image_type arg)
  {
    msg_.image = std::move(arg);
    return std::move(msg_);
  }

private:
  ::camera_interfaces::srv::GetImage_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::camera_interfaces::srv::GetImage_Response>()
{
  return camera_interfaces::srv::builder::Init_GetImage_Response_image();
}

}  // namespace camera_interfaces

#endif  // CAMERA_INTERFACES__SRV__DETAIL__GET_IMAGE__BUILDER_HPP_
