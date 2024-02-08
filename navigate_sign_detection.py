import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
import cv2
import numpy as np
import joblib
from geometry_msgs.msg import Point
from skimage.feature import hog
from skimage import exposure
import math
from cv_bridge import CvBridge

# States used in code
# 1 - GO Straight (followw_wall == False)
# 2 - Wall detected. Recognise the sign and make left or right turn
class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        image_qos_profile = QoSProfile(depth=5)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        self.navigation_cmd_publisher_ = self.create_publisher(
            Twist, '/cmd_vel', 10)
        
        self.visual_recognition_subscriber_ = self.create_subscription(
            String, '/sign', self.visual_recognition_callback, 10)
        
        self.lidar_subscriber_ = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, image_qos_profile)
        
        self._odom = self.create_subscription(
            Odometry, '/odom', self._odometry_callback, 1)

        self._video_subscriber = self.create_subscription(
            CompressedImage,
            # 'camera/image_raw/compressed',
            '/image_raw/compressed',
            self.visual_recognition_callback,
            image_qos_profile)

        self.latest_navigation_goal_ = None
        self.follow_wall_mode = True
        self.Init = True
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        self.globalAng = 0.0
        self.x_robot = 0
        self.y_robot = 0
        self.command = None
        self.desiredAng = 0.0
        self.obstacle_detected = False
        self.min=0.0
        self.heading_correction=0.0
        
        self.sign_detected = False
        self.loaded_knn = joblib.load('knn_model_weights.joblib')
        timer_period = 0.1 # Control or state machine update time (Digital Control)
        self.timer = self.create_timer(timer_period, self.stateMachine_callback)
        self.initial_heading = None
        self.range = 0.0

    def _odometry_callback(self, data):
        position = data.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = data.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
            self.initial_heading = self.Init_ang
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang
        self.globalAng = round(self.globalAng,4)

        self.x_robot = round(self.globalPos.x,3)
        self.y_robot = round(self.globalPos.y,3) 

    def stateMachine_callback(self):
        twist_msg = Twist()

        epsilon = 0.08

        if self.follow_wall_mode and not self.obstacle_detected:
            twist_msg.linear.x = 0.18
            if not math.isnan(self.heading_correction):
                twist_msg.angular.z = 10 * self.heading_correction
                print("------------ FORWARD ------------")
        
        else:
            heading_error = abs(self.wrapAngle(self.globalAng - self.desiredAng))

            if self.command == 'turn_left':
                if abs(heading_error) > epsilon:
                    twist_msg.angular.z = 0.25
                    print("------------ LEFT TURN ------------")

                else:
                    twist_msg.angular.z = 0.0
                    self.reset_state_flags()

            elif self.command == 'turn_right':
                if abs(heading_error) > epsilon:
                    twist_msg.angular.z = -0.25
                    print("------------ RIGHT TURN ------------")

                else:
                    twist_msg.angular.z = 0.0
                    self.reset_state_flags()

            elif self.command == 'uturn':
                if abs(heading_error) > epsilon:
                    twist_msg.angular.z = 0.4
                    print("------------ U-TURN ------------")

                else:
                    twist_msg.angular.z = 0.0
                    self.reset_state_flags()

            elif self.command == 'stop' or self.command == 'goal':
                print("------------ GOAL/STOP ------------")
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0

            elif self.command == 'unknown_command':
                print("------------ LOST  ------------>> ", self.min)
                print("------------ DISTANCE  ------------>> ", self.range[0])

                if self.min < 0.42:
                    self.handle_obstacle_avoidance(twist_msg)

                else:
                    self.command = 'turn_left'

        self.navigation_cmd_publisher_.publish(twist_msg)

    def reset_state_flags(self):
        twist_msg.angular.z = 0.0
        self.sign_detected = False
        self.follow_wall_mode = True
        self.command = None

    def handle_obstacle_avoidance(self, twist_msg):
        self.sign_detected = True
        if self.range[0] - self.min < 0.04:
            twist_msg.linear.x = -0.12
            twist_msg.angular.z = 0.0
            print("------------ REVERSE ------------")
            self.sign_detected = True

        else:
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.20
            print("------------ PERPENDICULAR MOVEMENT ------------")

    def scan_callback(self, data):
        ranges = data.ranges

        # Update minimum range
        self.min = min(ranges)
        self.range = ranges

        # Filter ranges to include only relevant data
        filtered_ranges = [ranges[i] for i in range(len(ranges)) if (0 <= i <= 10) or (350 <= i <= 359)]

        # Find the minimum range within the filtered range
        min_range = min(filtered_ranges)

        # Extract readings for left and right sectors
        right_readings = filtered_ranges[1:9]
        left_readings = filtered_ranges[-9:-1]

        # Calculate averages for left and right sectors
        avg_left = sum(left_readings) / len(left_readings)
        avg_right = sum(right_readings) / len(right_readings)

        # Calculate heading correction based on the difference between averages
        self.heading_correction = avg_right - avg_left

        # Update obstacle detection status
        if min_range < 0.55:
            self.obstacle_detected = True
            print("------------ DETECTED OBSTACLE ------------")
            self.follow_wall_mode = False
        else:
            self.obstacle_detected = False
        
    def visual_recognition_callback(self, msg):
        # Skip recognition when following the wall
        if self.follow_wall_mode:
            return

        # Proceed with recognition if no sign detected and not following wall
        if not self.sign_detected and not self.follow_wall_mode:
            # Convert sign to OpenCV format
            sign = CvBridge().compressed_imgmsg_to_cv2(msg, "bgr8")

            # Interpret the sign and generate command
            self.command = self.interpret_sign(sign)
            print("------------ COMMAND ------------>> ", self.command)
            print("------------ CURRENT HEADING ------------>> ", math.degrees(self.globalAng))
            print("------------ DESIRED HEADING ------------>> ", math.degrees(self.desiredAng))
    
    def interpret_sign(self, sign):
        # Convert sign to OpenCV format
        sign = CvBridge().compressed_imgmsg_to_cv2(sign, "bgr8")

        # Preprocess and crop the sign
        test_img = self.preprocess_and_crop(sign)

        # Default value for val_predictions
        val_predictions = 0

        # Process the sign if it's not None
        if test_img is not None:
            test_img = test_img.reshape(1, -1).astype(np.float32)
            val_predictions = self.loaded_knn.predict(test_img)[0]
            print(val_predictions)

        # Update sign detection status
        self.sign_detected = True

        # Interpret the sign and generate navigation command
        if val_predictions == 1.0:
            self.desiredAng = self.wrapAngle(self.globalAng + 1.52)
            return 'turn_left'
        elif val_predictions == 2.0:
            self.desiredAng = self.wrapAngle(self.globalAng - 1.52)
            return 'turn_right'
        elif val_predictions == 3.0:
            self.desiredAng = self.wrapAngle(self.globalAng - 3.14)
            return 'uturn'
        elif val_predictions == 5.0:
            return 'goal'
        elif val_predictions == 0.0 or val_predictions == 4.0:
            self.desiredAng = 0.0
            return 'unknown_command'
        
    def wrapAngle(self, angle):
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        wrapped_angle = (angle + math.pi) % (2 * math.pi) - math.pi
        print("wrapped = ", wrapped_angle)
        return wrapped_angle
    
    def preprocess_and_crop(self, image):
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the HSV ranges for red, green, and dark blue colors
        lower_red1 = np.array([0, 100, 100], dtype=np.uint8)
        upper_red1 = np.array([20, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([160, 100, 100], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        lower_green = np.array([40, 40, 40], dtype=np.uint8)
        upper_green = np.array([80, 255, 255], dtype=np.uint8)
        lower_dark_blue = np.array([110, 100, 20], dtype=np.uint8)
        upper_dark_blue = np.array([250, 255, 155], dtype=np.uint8)

        # Threshold the image to extract red, green, and dark blue colors
        mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
        mask_dark_blue = cv2.inRange(hsv_frame, lower_dark_blue, upper_dark_blue)

        # Combine the masks to get red, green, and dark blue regions
        mask_combined = cv2.bitwise_or(mask_red1, cv2.bitwise_or(mask_red2, cv2.bitwise_or(mask_green, mask_dark_blue)))
        mask = cv2.erode(mask_combined, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if contours were found
        if not contours:
            return None

        # Get the largest contour (assumed to be the road sign)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 50:
            buffer_size = 5

            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add a buffer to the bounding box
            x -= buffer_size
            y -= buffer_size
            w += buffer_size
            h += buffer_size

            # Create a bordered version of the image
            bordered_image = cv2.copyMakeBorder(image, buffer_size, buffer_size, buffer_size, buffer_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Ensure the adjusted bounding box stays within the bordered image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, bordered_image.shape[1] - x)
            h = min(h, bordered_image.shape[0] - y)

            # Draw the adjusted rectangle on the original image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the bordered image to the region of interest (ROI) with buffer
            cropped_sign = bordered_image[y:y+h+15, x:x+w+15]
            resized_img = cv2.resize(cropped_sign, (33, 25))

            enhanced_img = cv2.convertScaleAbs(resized_img, alpha=2.5, beta=5)
            cropped_sign_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

            hog_features, hog_image = hog(cropped_sign_gray, orientations=10, pixels_per_cell=(1, 1), cells_per_block=(1, 1), visualize=True)
            hog_features = hog_features.flatten()  # Flatten HOG features to ensure consistent shape
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

            # Display images (comment out if not needed)
            cv2.imshow('Original Image', hog_image_rescaled)
            cv2.imshow('Cropped Road Sign', enhanced_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return hog_features
        else:
            return None
        
    def show_image(self, img):
        cv2.imshow( "Raw Image", img)
        self._user_input = cv2.waitKey(50)
        if self._user_input == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit

def main(args=None):
    rclpy.init(args=args)

    navigation_node = NavigationNode()

    while rclpy.ok():
        rclpy.spin_once(navigation_node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()


