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
from geometry_msgs.msg import Pose2D

# SECCION DE IMPORTE DE LIBRERIAS ADICIONALES RELACCIONADAS A PYTHON
# Calculos matematicos y manejo matricial
import numpy as np
# Charts y graficacion
import matplotlib.pyplot as plt
import matplotlib
import json
# Elements de analisis de alto nivel
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from scipy import stats
matplotlib.use('TkAgg')

# Inicializacion de objeto padre de procesamiento
class DetectObjects(Node):
    ruta_almacenamiento = '/home/elio/ws_ros/archivos_v'
    def __init__(self):
        super().__init__('obstacle_scan')
        # Inicializacion de las variables
        self._init_variables()
        # Inicializacion de los subscriptorers
        self._init_subs()
        # Inicializacion de publicadores
        self._init_publisher()

        # Inizacion del timer de ejecucion del programa de lectura
        self.timer = self.create_timer(0.5, self.loop) # Toma de datos cada 0.4 sg

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
        self.types_objects = 1 # int(input("Objetos Grades: 0, Objetos Pequeños, 1: "))
        self.val_temp = False
        self.posicion_objetivo = Pose2D()
        self.posicion_objetivo.theta = 0.0
        self.lista_objetos = []
        self.ruta_archivo = 'Lista_objetos.json'
        self.ruta_archivo_f = self.__class__.ruta_almacenamiento + '/' + self.ruta_archivo
        self.datos_json = self.cargar_datos(self.ruta_archivo_f)

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
    
    def _init_publisher(self):
        self.publicar_pos_ob = self.create_publisher(
            Pose2D,
            '/pos_objeto',
            10
        )
    
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
        room_points, index = self.calculate_cartesian_coordinates(scan_data, robot_position, robot_orientation)
        self.room_points = room_points[:]
        
        #print(self.room_points)
        # Se aplica la funcion de clusterizacion de DBSCAN para obtener la posicion estimada de los obstaculos
        R_value, puntos_agrupados = self.func_id_obstacles(
                vec_valores = room_points,
                tipo_escala = self.types_objects
            )
        
        #print(R_value)
        # Se extrae los puntos de corresponden a obstaculos y los que no corresponden a obstaculos
        try:
            array_original_str = np.array([str(i) for i in room_points])
            array_extraido_str = np.array([str(i) for i in np.concatenate(puntos_agrupados)])
            # Se obtiene la mascada de puntos de 
            mascara = np.isin(array_original_str, array_extraido_str, invert=True)
            # Array filtrado de solo objetos pertenecientes al mapa
            array_filtrado = room_points[mascara]
        except Exception as e:
            array_filtrado = room_points
            self.get_logger().info("Error en la lectura de objetos")
    

        # Publicacion del mapa
        self.ax.scatter(array_filtrado[:, 0], array_filtrado[:, 1], marker = '.', color = 'g')
        # Publicacion de obstaculos
        lista_cen_temp, lista_vecs_s = [], []
        for num, vec in enumerate(puntos_agrupados):
            try:
                # Seccion de calculo de centroide
                x_coords = [p[0] for p in vec]
                y_coords = [p[1] for p in vec]
                centroid = [sum(x_coords) / len(vec), sum(y_coords) / len(vec)]
                distances = [np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2) for x, y in vec]
                punto_A = [x_coords[0], y_coords[0]]
                #print(f"centroid : {centroid}, Distancias : {distances}")
                radius = np.mean(distances)
                # val = all(abs(d - radius) < 0.1 for d in distances)
                #if not self.val_temp:
                # lista_cen_temp.append(punto_A)
                #self.posicion_objetivo.x = centroid[0] #+ robot_position[0]
                #self.posicion_objetivo.y = centroid[1] #+ robot_pos ition[1]
                # self.posicion_objetivo.x = x_coords[len(x_coords)//2]
                # self.posicion_objetivo.y = y_coords[len(y_coords)//2]
                #self.publicar_pos_ob.publish(self.posicion_objetivo)
                #self.val_temp = True
                # Calculo posible cuadrado

                # if val == True:
                #print(f"robot_position = {robot_position}, obstaculo = {centroid}")
                distance_r_c = np.sqrt((robot_position[0] - centroid[0])**2 + (robot_position[1] - centroid[1])**2)
                
                if distance_r_c < 1.8 and distance_r_c > 0.4:
                #if distance_r_c < 10 and distance_r_c > 0.0:
                    #print(f"objeto {num} = {R_value[num]}")
                    #lista_vecs_s.append(vec)
                    lista_cen_temp.append(centroid)
                    self.ax.text(centroid[0], centroid[1], f"Objeto", color = "black")
                    self.ax.scatter(vec[:, 0], vec[:, 1], marker = '.', color = 'r')
                    self.ax.scatter(centroid[0], centroid[1], marker = '.', color = 'r')
                else:
                    self.ax.scatter(vec[:, 0], vec[:, 1], marker = '.', color = 'g')
                
                # else:
                #     self.ax.text(centroid[0], centroid[1], "P_Cuadrado", color = "indigo")
                #     #self.ax.scatter(vec[:, 0], vec[:, 1], marker = '.', color = 'r')
                #     self.ax.scatter(centroid[0], centroid[1], marker = '.', color = 'r')

            
            except Exception as e:
                self.get_logger().info("No se detectaron objetos")
            #try:
            # if self.is_circle(self, vec):   
            #     centroide_figura = self.calculate_centroid(self, vec)
            #     self.get_logger().info(centroide_figura)
                    #self.ax.text(centroide_figura[0], centroide_figura[1], "Circle", color = "black")
            
            # except Exception as E:
            #     print(E)
            #     self.get_logger().info("No se detectaron objetos")

        if len(lista_cen_temp) > 0:
            #print(len(lista_cen_temp) == len(lista_vecs_s))
            self.update_centroid_points(np.array(lista_cen_temp))
        #print(self.lista_objetos)
        #print(self.datos_json)
        self.guardar_datos(self.ruta_archivo_f, self.datos_json)

        self.ax.scatter(robot_position[0], robot_position[1], marker = 'H', color = 'b')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        # self.ax.set_xlim(robot_position[0] -4, robot_position[0] + 4)
        # self.ax.set_ylim(robot_position[1] -4, robot_position[1] + 4)
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')

        plt.draw()
        plt.pause(0.001)

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

    # Funcion de deteccion de objetos
    # Funcion para la identificacion de obstaculos
    # Procesamiento de deteccion de obstaculos
    # Algoritmo de identificacion de agrupamiento de puntos -> score de alineamiento del modelo de regresion lineal
    # vec_valores -> Corresponde al vector de los valores a identificar
    # distancia_entre_puntos -> Distancia entre puntos deseada
    # cant_min_puntos -> Cantidad minima de puntos para la agrupacion de cada subvector
    # threshold  -> Score maximo de refferencia de pared
    def func_id_obstacles(
            self, 
            vec_valores: np.array,
            distancia_entre_puntos: float = 0.3,
            cant_min_puntos: float = 3,
            threshold: float = 1-0.1,
            tipo_escala: int = 0,
            tam_max: int = 80,
            tam_min: int = 10
        ):
        lista_r = []
        # Identificador de tipos de objetos
        if tipo_escala == 1:
            tam_max = 40
            tam_min = 5
        # threshold = 1 # Valor de treshold dado para un muro
        # distancia_entre_puntos = 0.3 # Un valor arbitraria para determinar la distancia maxima entre los puntos a evaluar
        # cant_min_puntos = 3 # numero minimo de puntos para realizar la agrupacion
        clust = DBSCAN(eps = distancia_entre_puntos, min_samples = cant_min_puntos).fit(vec_valores) # Calculo de clustering
        IDX = clust.labels_ # Extraccion de labels
        # self.get_logger().info(f'{len(IDX)}')
        k = np.max(IDX) # Eleccion de k maxima
        lista_identificados = []
        for i in range(k + 1): # Por cada valor 
            
            o_ = vec_valores[IDX == i]
            
            if o_.size == 0: # Si no se encuentran valores continue con el siguiente valor de IDX
                continue
            if o_.size >= 3: # Si tiene mas de 3 valores
                # Realiza la regresión lineal -> Sistema de clasificacion por regresion lineal
                slope, intercept, r_value, p_value, std_err = stats.linregress(o_[:, 0], o_[:, 1]) # R_value corresponde al valor de R
                r_value_q = r_value**2
                if r_value_q:
                    lista_r.append(r_value_q)
                
                #print(o_.size)
                #print(f"k = {i}, value = {r_value_q}")
                if r_value_q >= threshold:
                    pass
                elif r_value_q > 0.08 and r_value_q < 1:
                    if o_.size < tam_max and o_.size > tam_min:
                        #print(f"gurpo {i}= {r_value_q}")
                        lista_identificados.append(o_)
                else:
                    pass

        return lista_r, lista_identificados
    
    # Temporal se puede desagregar
    # Funcion para deteccion de circulos
    def is_circle(self, points, tolerance= 0.1):
        centroid = self.calculate_centroid(points)
        distances = [np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2) for x, y in points]
        radius = np.mean(distances)
        return all(abs(d - radius) < tolerance for d in distances)
    
    # Calculo de funcion de centroide
    def calculate_centroid(self, points):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        centroid = (sum(x_coords) / len(points), sum(y_coords) / len(points))
        return centroid
    
    # actualizacion de punto centroide de objetos
    def update_centroid_points(self, new_points, threshold = 0.8):
        #if new_points != None:
        for point in new_points:
            if not any(np.linalg.norm(point - existing_point) < threshold for existing_point in self.lista_objetos):
                self.lista_objetos.append(point)

                n_objeto = f"Objeto_{len(self.lista_objetos)}"

                self.datos_json[n_objeto] = {'Coordenada': point.tolist()}

    #SECCION DE FUNCION UTILITARIAS
    def cargar_datos(self, ruta_archivo):
        try:
            with open(ruta_archivo, 'r') as archivo:
                datos = json.load(archivo)
        except FileNotFoundError:
            datos = {}  # Si el archivo no existe, crea un diccionario vacío
        return datos

    # Función para guardar datos en un archivo JSON
    def guardar_datos(self, ruta_archivo, datos):
        with open(ruta_archivo, 'w') as archivo:
            json.dump(datos, archivo, indent=4)
    
def main(args=None):
    rclpy.init(args=args)
    Objetos_detect = DetectObjects()
    rclpy.spin(Objetos_detect)
    Objetos_detect.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()