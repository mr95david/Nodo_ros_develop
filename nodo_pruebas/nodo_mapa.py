#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf_transformations as tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy import stats
matplotlib.use('TkAgg')

class cartografo_propio(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_scan')
        self._init_variables()
        self._init_subs()
        self.timer = self.create_timer(0.4, self.loop)
        self.fig_2 = plt.figure(figsize=(10, 10))
        x, y = 1000, 100
        self.fig_2.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        self.ax = self.fig_2.add_subplot(111)
        plt.show(block=False)
        self.get_logger().info("Node started")
    
    def _init_variables(self):
        self.robot_pose = None
        self.scan_data = None
        self.l_max = 10.0
        self.l_min = -10.0
        self.l_0 = 0.0 
        self.lo_b = 0.5
        self.lo_o = 0.3
        self.epsilon_alpha = 0.1
        self.epsilon_d = 0.1
        self.grid_size = 105
        self.l = np.full((self.grid_size, self.grid_size), self.l_0)
        self.x_offset = 0 
        self.y_offset = 0  

    def _init_subs(self):
        self.scan_sub = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.scan_callback, 
            10)
        self.pose_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            self.pose_callback, 
            10)

    def scan_callback(self, scan_data: LaserScan):
        self.scan_data = scan_data
        self.max_range = self.scan_data.range_max
        self.d_max = self.max_range
        self.num_laser_points = len(scan_data.ranges)
    
    def pose_callback(self, pose_data):
        self.robot_pose = pose_data.pose.pose

    def loop(self):
        if self.robot_pose is None or self.scan_data is None:
            return
        self.ax.clear()
        scan_data = self.scan_data
        robot_position = [self.robot_pose.position.x, self.robot_pose.position.y]
        robot_orientation = self.robot_pose.orientation
        
        room_points, index = self.calculate_cartesian_coordinates(scan_data, robot_position, robot_orientation)

        z = room_points[:]
        q = [robot_position[0], robot_position[1], robot_orientation]
        self.l = self.update_log_odds(
            self.l, z, q, self.d_max, self.epsilon_alpha,
            self.epsilon_d, self.l_max, self.l_min,
            self.l_0, self.lo_b, self.lo_o)
        
        self.ax.imshow(self.l, cmap='Greys', interpolation='nearest')
        # plt.colorbar()  # Show the color scale

        # In a heatmap, the y-axis is inverted by default (i.e., the [0,0] index is the top-left corner of the plot).
        # If you want the [0,0] index to be at the bottom-left corner, you can invert the y-axis:
        #plt.gca().invert_yaxis()

        #plt.show()
        #self.ax.imshow(self.l, cmap='hot', interpolation='nearest')
        self.ax.invert_yaxis()
        plt.draw()
        plt.pause(0.001)

    def calculate_cartesian_coordinates(self, scan_data, robot_position, robot_orientation):
        # Angulo minimo de medida del sensor LIDAR
        current_angle = scan_data.angle_min

        # Almacenamiento de posiciones en el mapa de acuerdo a la transformacion euclidiana
        x = []
        y = []
        index = []
        # Matriz de rotacion
        rotation_matrix = tf.quaternion_matrix([robot_orientation.x,
                                                robot_orientation.y,
                                                robot_orientation.z,
                                                robot_orientation.w])
        # Identificacion de balores y especificacion de datos 
        for idx, r in enumerate(scan_data.ranges):
            if r < scan_data.range_min or r > scan_data.range_max:
                current_angle += scan_data.angle_increment
                continue
            local_x = r * np.cos(current_angle)
            local_y = r * np.sin(current_angle)
            # Salida dedl punto en el mundo
            point_in_world = np.dot(rotation_matrix, [local_x, local_y, 0, 1])[:3]
            world_x = point_in_world[0] + robot_position[0]
            world_y = point_in_world[1] + robot_position[1]
            # Agregar los valores obtenidos en las listas de posicionamiento
            x.append(world_x)
            y.append(world_y)
            index.append(idx)
            # Angulo actual del Scan
            current_angle += scan_data.angle_increment
        return np.array([x, y]).T, index

    def update_log_odds(self, l, z, q, d_max, epsilon_alpha, epsilon_d, l_max, l_min, l_0, lo_b, lo_o):
        q_x, q_y, q_theta = q
        for x, y in z:
            x_adjusted = int(round(x*10, 0)) + self.x_offset
            y_adjusted = int(round(y*10, 0)) + self.y_offset
            r = np.sqrt((x_adjusted - q_x)**2 + (y_adjusted - q_y)**2)
            l, new_x, new_y = self.expand_grid_if_necessary(l, x_adjusted, y_adjusted)
            if r <= d_max:
                if l[new_y][new_x] < l_max:
                    l[new_y][new_x] += lo_o
            else:
                if l[new_y][new_x] > l_min:
                    l[new_y][new_x] -= lo_b
        return l

    def expand_grid_if_necessary(self, l, x, y):
        y_max, x_max = l.shape
        x_offset, y_offset = 0, 0
        if x < 0:
            l = np.hstack((self.l_0 * np.ones((y_max, abs(x))), l))
            x_offset = abs(x)
        elif x >= x_max:
            l = np.hstack((l, self.l_0 * np.ones((y_max, x - x_max + 1))))
        if y < 0:
            l = np.vstack((self.l_0 * np.ones((abs(y), x_max + x_offset)), l))
            y_offset = abs(y)
        elif y >= y_max:
            l = np.vstack((l, self.l_0 * np.ones((y - y_max + 1, x_max + x_offset))))
        self.x_offset += x_offset
        self.y_offset += y_offset
        return l, x + self.x_offset, y + self.y_offset

def main(args=None):
    rclpy.init(args=args)
    draw_map = cartografo_propio()
    rclpy.spin(draw_map)
    draw_map.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()