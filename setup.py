from setuptools import find_packages, setup

package_name = 'object_detection_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/detection_launch.py']),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nahian',
    maintainer_email='nahianprf121@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['rosbot_camera_subscriber = object_detection_pkg.rosbot_camera_subscriber:main',
        'object_detection_node = object_detection_pkg.object_detection_node:main',
        ],
    },
)