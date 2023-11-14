#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
import time 

from arm05_sim.action import Countdown

class CountdownServer(Node):
    def __init__(self):
        super().__init__("countdown_server")
        self._action_server = ActionServer(
            self,
            Countdown,
            "countdown",
            self.execute_callback)
# Callback function to run after acknowledging a goal from the client
    def execute_callback(self, goal_handle):
        self.get_logger().info("Starting countdown…") \

        feedback_msg = Countdown.Feedback()
        feedback_msg.current_num = goal_handle.request.starting_num

        while feedback_msg.current_num > 0:
	        # Decrement the feedback message’s current_num
            feedback_msg.current_num = feedback_msg.current_num - 1

           # Print log messages
            self.get_logger().info('Feedback: {0}'.format(feedback_msg.current_num))
            goal_handle.publish_feedback(feedback_msg)

	        # Wait a second before counting down to the next number
            time.sleep(1)

        # Indicate that the goal was successful
        goal_handle.succeed()
        result = Countdown.Result()
        result.is_finished = True
        return result

def main(args=None):
    rclpy.init(args=args)
    countdown_server = CountdownServer()
    rclpy.spin(countdown_server)

if __name__ == '__main__':
    main()



