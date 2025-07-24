import rclpy
from rclpy.node import Node
from pop.driving import Driving

class DrivingParam(Node):
    TIMER_PERIOD = 0.03
    def __init__(self):
        super().__init__('driving_param')
        self.drv = Driving()
        self.declare_parameter("steering",0)
        self.declare_parameter("direction",0)
        self.declare_parameter("throttle",0)
        self.steering_timer = self.create_timer(self.TIMER_PERIOD,self.steering_cb)
        self.move_timer = self.create_timer(self.TIMER_PERIOD,self.move_cb)        

    def steering_cb(self):
        steering = self.get_parameter("steering").get_parameter_value ().double_value
        self.drv.steering = steering

    def move_cb(self):
        direction = self.get_parameter("direction").get_parameter_value().integer_value
        throttle = self.get_parameter("throttle").get_parameter_value().integer_value
        if direction == 1:
            self.drv.forward(throttle)
        elif direction == 2:
            self.drv.backward(throttle)
        elif direction == 3:
            self.drv.stop()

def main(args=None):
    rclpy.init(args=args)
    driving_param = DrivingParam()
    rclpy.spin(driving_param)
    rclpy.shutdown()

if __name__=='__main__':
    main()
