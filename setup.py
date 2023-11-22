from setuptools import find_packages, setup

package_name = 'nodo_pruebas'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='elio',
    maintainer_email='eliotrianar95@gmail.com',
    description='Assignment 1: Autonomous Robots',
    license='License: Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [    
            'detect_objects = nodo_pruebas.nodo_objetos:main',
            'dibujar_mapa = nodo_pruebas.nodo_mapa:main',
            'mapa_2 = nodo_pruebas.mapa_2:main',
            'mov_bot = nodo_pruebas.prueba_gaz_3_3:main'
        ],
    },
)
