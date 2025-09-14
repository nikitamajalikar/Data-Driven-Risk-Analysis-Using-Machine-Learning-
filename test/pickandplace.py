from envs.Locate import Locate
from envs.panda_transformer import PandaTransformer
import numpy as np
from hw_interface import HwInterface  # Assume the HwInterface class is in a file named hw_interface.py
import matplotlib.pyplot as plt
import random
import json
from logger import JointLogger

# Initialize the hardware interface with panda, camera, and gazebo enabled
hw_interface = HwInterface(panda=True, cameras=["/image_raw"], gazebo=True)
log=JointLogger
# Test get_component_info method
components = hw_interface.get_component_info()
# print("Components:", components)

# Test get_robot_position method
robot_position = hw_interface.get_robot_position()
print("Robot Position:", robot_position)

# Test get_gripper_position method
gripper_position = hw_interface.get_gripper_position()
print("Gripper Position:", gripper_position)

my_trans = PandaTransformer()
my_trans.init_pose()
locate = Locate()
base_coordinates, relative_position = locate.locate_colored_object(colormode='cube')

results = {}
velocity_array=[0.58057806, 0.71882371, 0.42048698, 0.74302886, 0.40293072,
       0.20449751, 0.45862464, 0.58266626, 0.27037792, 0.27928531,
       0.6310157 , 0.39124002, 0.44795776, 0.88055234, 0.24424888,
       0.55990841, 0.46262567, 0.28368923, 0.49072921, 0.356561 ]
print(velocity_array[3])
print(len(velocity_array))
coordinates = np.empty((4, 7), dtype=float)
arr = []

#hw_interface.publish_pose()
print('test')

for x in range(len(velocity_array)):
    print('#########################################################################################################')
    print(x)
    print('#########################################################################################################')
    robot_position_1 = hw_interface.get_robot_position()
    print('robot_position_1', robot_position_1)
    #coordinates= np.append(coordinates, robot_position_1, axis=0)
    arr.append([robot_position_1])

    pick_coordinates, pick_relative_position = locate.locate_colored_object(colormode='cube')
    print("Object position:", base_coordinates)

    # Open the gripper
    hw_interface.move_gripper()

    # Move the gripper to the detected position
    # x = 0.486 y=0.009
    success_move = hw_interface.move_cartesian(x=pick_coordinates[0], y=pick_coordinates[1], z=0.2, vel=velocity_array[x], acc=0.5)
    # success = hw_interface.move_cartesian(x=0.486, y=0.009, z=0.2, vel=0.5, acc=0.5)
    print("Move Cartesian:", "Success" if success_move else "Failed")

    robot_position_2 = hw_interface.get_robot_position()
    print('robot_position_2', robot_position_2)
    #coordinates=np.append(coordinates, robot_position_2, axis=0)
    arr.append([robot_position_2])

    # Grasp the object
    hw_interface.move_cartesian(z=0.025, vel=velocity_array[x], acc=0.5)
    success_pick = hw_interface.grasp_object(width=0.03)
    print("Pick:", "Success" if success_pick else "Failed")

    # Move the gripper up after grasping
    hw_interface.move_cartesian(z=0.3, vel=velocity_array[x], acc=0.5)

    # Move to random position
    x_pos = random.uniform(-0.1, 0.6)
    y_pos = random.uniform(-0.2, 0.3)
    hw_interface.move_cartesian(x=x_pos, y=y_pos, vel=velocity_array[x], acc=0.5)

    robot_position_3 = hw_interface.get_robot_position()
    print('robot_position_3', robot_position_3)
    #coordinates=np.append(coordinates, robot_position_3, axis=0)
    arr.append([robot_position_3])

    # Place the object
    hw_interface.move_cartesian(x=base_coordinates[0], y=base_coordinates[1], vel=0.5, acc=0.5)
    hw_interface.move_cartesian(z=0.025, vel=velocity_array[x], acc=0.5)
    hw_interface.move_gripper()

    # go back to init
    hw_interface.move_cartesian(z=0.2, vel=0.5, acc=0.5)
    # my_trans.init_pose()
    hw_interface.move_cartesian(x=0.3, y=0.0, z=0.5, vel=velocity_array[x], acc=0.5)

    robot_position_4 = hw_interface.get_robot_position()
    print('robot_position_4', robot_position_4)
    #coordinates=np.append(coordinates, robot_position_4, axis=0)
    arr.append([robot_position_4])

    # check if object is successfully placed at the same position
    place_coordinates, place_relative_position = locate.locate_colored_object(colormode='cube')
    x_deviation = abs(base_coordinates[0] - place_coordinates[0])
    print('X deviation: ', x_deviation)
    y_deviation = abs(base_coordinates[1] - place_coordinates[1])
    print('Y deviation: ', y_deviation)

    if x_deviation<=0.01 and y_deviation<=0.01:
        y = 1
        print('Task successful')
    else:
        y = 0
        print('Task failed')

    results[x] = y

with open('vel_results.json', 'w') as outfile:
    json.dump(results, outfile, indent=4)

np_arr = np.array(arr)
np.save('vel_robot_pos.npy', np_arr)

