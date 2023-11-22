#!/usr/bin/env python3

# El siguiente nodo busca realiar la deteccion e identificacion de objetos 
# en un mapa cualqueria a partir de la lectura de un sensor LIDAR, determinando
# las caracteristicas de cambio de cada objeto medido... -> agregar tareas adicionales

# SECCION DE IMPORTE DE LIBRERIAS DE ROS2
import rclpy
from rclpy.node import Node
import tf_transformations as tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# SECCION DE IMPORTE DE LIBRERIAS ADICIONALES RELACCIONADAS A PYTHON
# Calculos matematicos y manejo matricial
import numpy as np
# Charts y graficacion
import matplotlib.pyplot as plt
import matplotlib
# Elements de analisis de alto nivel
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy import stats
matplotlib.use('TkAgg')

# Inicializacion de objeto padre de procesamiento
class DetectObjects(Node):
    def __init__(self):
        super().__init__('obstacle_scan')
        # Inicializacion de las variables
        self._init_variables()
        # Inicializacion de los subscriptorers
        self._init_subs()

        # Inizacion del timer de ejecucion del programa de lectura
        self.timer = self.create_timer(0.4, self.loop) # Toma de datos cada 0.4 sg

        # Seccion de inicio de proceso de gracficion
        self.fig = plt.figure(figsize=(10, 10))
        x, y = 1000, 100
        self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))

        self.ax = self.fig.add_subplot(111)
        # Muestra de grafica
        plt.show(block=False)
        self.get_logger().info("Node started")

    # Funcion de inicializacion de varibles
    def _init_variables(self):
        self.robot_pose = None # Variable de almacenamiento del robot
        self.scan_data = None # Variable de almacenamiento de las medidas
        # self.types_objects = int(input("Objetos Grades: 0, Objetos Pequeños, 1: "))
        self.room_points_s = []

    # Funcion de inicializacion de variables
    def _init_subs(self):
        # Subscripcion para la lectura del sensor.
        self.scan_sub = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.scan_callback, 
            10)
        # Subscripcion para la lectura de la odometria
        self.pose_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            self.pose_callback, 
            10)
    
    # SEccion de callbacks de subscriptores
    # Lectura de sensor lidar
    def scan_callback(self, scan_data):
        self.scan_data = scan_data
        self.num_laser_points = len(scan_data.ranges)
    
    # Lectura de pose actual
    def pose_callback(self, pose_data):
        self.robot_pose = pose_data.pose.pose

    # Seccion de ejecucion principal del nodo
    def loop(self):
        # Si la posicion del robot no esta definida, rompe la funcion
        if self.robot_pose is None or self.scan_data is None:
            return

        # Borra la imagen actual del ax de la figura del mapa
        self.ax.clear()
        # Obtiene en una varibale local la informacion obtenida por el lase
        scan_data = self.scan_data
        # Realzia un a lista de la posicion actual del robot
        robot_position = [self.robot_pose.position.x, self.robot_pose.position.y]
        robot_orientation = self.robot_pose.orientation
        
        # Obtiene las posiciones de los obstaculos
        new_room_points, _ = self.calculate_cartesian_coordinates(scan_data, robot_position, robot_orientation)
        
        #print(new_room_points)
        self.update_room_points(new_room_points)
        if self.room_points_s:
            plot_room_points = np.array(self.room_points_s)
            #print(plot_room_points)
            self.ax.scatter(plot_room_points[:, 0], plot_room_points[:, 1], marker = 'H', color = 'b')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_aspect('equal')

        plt.draw()
        plt.pause(0.001)
        #print(self.room_points_s)

    def update_room_points(self, new_points, threshold = 0.2):
        #if new_points != None:
        for point in new_points:
            if not any(np.linalg.norm(point - existing_point) < threshold for existing_point in self.room_points_s):
                self.room_points_s.append(point)

    # Seccion de funciones utilitarias
    # Funcion para el calculo de la posicion en el plano cartesiano
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
    
def main(args=None):
    rclpy.init(args=args)
    Objetos_detect = DetectObjects()
    rclpy.spin(Objetos_detect)
    Objetos_detect.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

