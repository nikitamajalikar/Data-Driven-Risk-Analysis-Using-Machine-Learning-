import rospy
from geometry_msgs.msg import Pose
import numpy as np
from hw_interface import HwInterface  # Assume the HwInterface class is in a file named hw_interface.py
import matplotlib.pyplot as plt

import json
from scipy.stats import alpha, beta, lognorm
from envs.Locate import Locate
from envs.panda_transformer import PandaTransformer
#from hw_interface_test_combination_nikita import success_rate, failure_rate

# Initialize the hardware interface with panda, camera, and gazebo enabled
hw_interface = HwInterface(panda=True, cameras=["/image_raw"], gazebo=True)

# Test get_component_info method
components = hw_interface.get_component_info()
print("Components:", components)

# Test get_robot_position method
robot_position = hw_interface.get_robot_position()
print("Robot Position:", robot_position)

# Test get_gripper_position method
gripper_position = hw_interface.get_gripper_position()
print("Gripper Position:", gripper_position)

# Initialize the transformer and locate object
my_trans = PandaTransformer()
my_trans.init_pose()
locate = Locate()
base_coordinates, relative_position = locate.locate_colored_object(colormode='cube')
print("Object position:", base_coordinates)

# List to store data
data_log = {
   "initial_positions": {
       "robot_position": robot_position,
       "object_position": base_coordinates,
       "gripper_position": gripper_position,
   },
   "iterations": []
}


Sampled_Position_X = np.array([0.50162274, 0.48951069, 0.50341215, 0.49615727, 0.48101159, 0.47225563,
 0.501272,   0.48676327, 0.46111212, 0.47396221, 0.48636629, 0.49040326,
 0.49434653, 0.48770368, 0.49803126, 0.48269194, 0.4861956,  0.47861138,
 0.4871818,  0.47833452, 0.49097439, 0.49541848, 0.49616701, 0.49587376,
 0.48589114, 0.48176583, 0.50880319, 0.50292152, 0.49137836, 0.48756908,
 0.49160862, 0.49258558, 0.4923422,  0.5045687,  0.48087306, 0.50729335,
 0.49387115, 0.48184728, 0.48098471, 0.48100386, 0.47912577, 0.48962986,
 0.49796531, 0.49139124, 0.50437722, 0.47272433, 0.46046471, 0.50886724,
 0.47090171, 0.48231243])
Sampled_Position_Y = np.array([0.00583692, -0.00183908,  0.01022704,  0.01272923,  0.01383809, -0.00853211,
 -0.00863691,  0.00434802,  0.0148004,  -0.0062895,  -0.00249842,  0.01237284,
  0.01789194,  0.00260495,  0.00278459,  0.01536117, -0.00491076,  0.00727189,
 -0.01302764, -0.00467967, -0.0046506,   0.00556309, -0.00939853,  0.00011278,
  0.01854924,  0.00540417,  0.00649814,  0.02043808, -0.00225368, -0.00759682,
  0.00524199,  0.01498114, -0.00356128, -0.00178039,  0.01685438,  0.00172228,
 -0.00232857,  0.0112997,   0.01051437, -0.00626076,  0.00211887,  0.01688476,
  0.00964298,  0.00292617,  0.0043659,  -0.00725703,  0.01913036,  0.01320721,
  0.00547154, -0.00220461])
velocities = np.array([0.42703107, 0.65054628, 0.92381211, 0.88242321, 0.06409533, 0.51991044,
 0.43203649, 0.27483223, 0.83336267, 0.02550827, 0.75090269, 0.02264429,
 0.65405846, 0.66280533, 0.56989851, 0.68723757, 0.86994983, 0.36043064,
 0.34402646, 0.65436021, 0.6591422,  0.78463515, 0.88210962, 0.72336961,
 0.25443845, 0.76279022, 0.46886211, 0.36765262, 0.59939354, 0.30912722,
 0.66781543, 0.30594075, 0.06743806, 1.00547776, 0.391729,   0.3916542,
 0.37883287, 0.7622545,  0.46375612, 0.5830149,  0.69778815, 0.35644257,
 0.13074546, 0.04309196, 0.35561919, 0.0730527,  0.12520682, 0.14678731,
 0.3663727,  0.76891574])
accelerations = np.array([0.0467997,  0.35615545, 0.17009115, 0.24398892, 0.4896305,  0.01974982,
 0.94418495, 0.43542593, 0.96057379, 0.08151943, 0.23918518, 0.97230289,
 0.94863992, 0.92756688, 0.03830152, 0.24073922, 0.74912436, 0.48310211,
 0.46059096, 0.39073863, 0.51503254, 0.30436251, 0.3912778, 0.46240073,
 0.53232177, 0.28667387, 0.08246239, 0.96527791, 0.83631219, 0.31838201,
 0.25881522, 0.25272922, 0.48158767, 0.93534967, 0.39018217, 0.37232855,
 0.54877344, 0.21277038, 0.8446034,  0.3901857,  0.8445308,  0.60013632,
 0.19664784, 0.19792021, 0.66873008, 0.40150629, 0.82411241, 0.2059014,
 0.66412756, 0.33099307])

gripper_width = np.array([0.02684837, 0.0234594,  0.03962826, 0.0201221,  0.02268935, 0.02526456,
 0.03840064, 0.02629163, 0.03076169, 0.03627818, 0.03557461, 0.03887616,
 0.03899043, 0.02357946, 0.02643949, 0.03110151, 0.02843104, 0.02843928,
 0.03842656, 0.03544561, 0.03413455, 0.02764091, 0.02018796, 0.03905807,
 0.03845678, 0.02295339, 0.03321773, 0.029642,   0.02133097, 0.03390705,
 0.02423955, 0.02086985, 0.03394813, 0.02635472, 0.02848267, 0.0318311,
 0.03684087, 0.03400123, 0.02787337, 0.02904743, 0.03712919, 0.03479383,
 0.03875053, 0.03995893, 0.02000533, 0.03913015, 0.02350454, 0.02574158,
 0.02528663, 0.03632591])

# Lists to store iteration data for plotting
iteration_numbers = []
selected_positions = []
grasp_results = []

for i in range(50):
    # Select velocity for this iteration
    vel = velocities[i]
    acc = accelerations[i]
    gw =gripper_width[i]
    generated_X = Sampled_Position_X [i]
    generated_Y = Sampled_Position_Y [i]

    print(f"Iteration: {i + 1}, Selected position for action: x={generated_X}, y={generated_Y}, Velocity {vel}, Acceleration {acc}, Gripper width {gw}")
    iteration_data = {
        "iteration": i + 1,
        "selected_position": {"x":generated_X , "y": generated_Y},
        "velocity": vel,
        "accelerations": acc,
        "Gripper width": gw,
        "actions": []
    }

    # Open the gripper
    hw_interface.move_gripper()
    iteration_data["actions"].append({"action": "move_gripper", "status": "opened"})

    # Test move_cartesian method with the selected velocity
    success = hw_interface.move_cartesian(x=generated_X, y=generated_Y, z=0.2, o_x=1, o_y=0, o_z=0, o_w=0, vel=vel, acc=acc)
    print("Move Cartesian:", "Success" if success else "Failed")
    hw_interface.move_cartesian(z=0.02, vel=vel, acc=acc)

    # Test grasp_object method
    success = hw_interface.grasp_object(width=gw)
    iteration_data["actions"].append(
       {"action": "grasp_object", "status": "Success" if success else "No object grasped"})
    print("Grasping object:", "Success" if success else "No object grasped")

    # Test place object in another coordinate
    hw_interface.move_cartesian(z=0.2, vel=vel, acc=acc)
    hw_interface.move_cartesian(x=0.6, y=0.05, vel=vel, acc=acc)
    hw_interface.move_cartesian(x=generated_X, y=generated_Y, vel=vel, acc=acc)
    hw_interface.move_cartesian(z=0.02, vel=vel, acc=acc)
    hw_interface.move_gripper()
    hw_interface.move_cartesian(z=0.2, vel=vel, acc=acc)
    hw_interface.move_cartesian(x=0.2, y=0.1, vel=vel, acc=acc)

    # Append iteration data to data log
    data_log["iterations"].append(iteration_data)

    # Collect data for plotting
    iteration_numbers.append(i + 1)
    selected_positions.append((generated_X, generated_Y))
    grasp_results.append(success)

# Convert NumPy arrays to lists for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

# Calculate success and failure rates
success_rate = sum(grasp_results) / len(grasp_results)
failure_rate = 1 - success_rate
print(f"Success Rate: {success_rate * 100:.2f}%")
print(f"Failure Rate: {failure_rate * 100:.2f}%")

# Save data to JSON file with success and failure rates
data_log["summary"] = {
    "success_rate": success_rate,
    "failure_rate": failure_rate}
# Save the log data to a JSON file
with open("hw_interface_log_combination.json", "w") as f:
   json.dump(data_log, f, indent=4, default=convert_to_serializable)


# Create iteration numbers (sample numbers)
#iteration_numbers = np.arange(1, len(velocities) + 1)
#iteration_numbers = np.arange(1, len(accelerations) + 1)
# Define colors based on grasp_results: green for success, red for failure
#colors = ['green' if success else 'red' for success in grasp_results]

# Create the scatter plot
#plt.figure(figsize=(10, 6))
#plt.scatter(iteration_numbers, accelerations, color=colors, label='Acceleration Samples', marker='o')

# Add title and labels
#plt.title(f"Scatter Plot of Acceleration vs Sample Number\n\nSuccess Rate: {success_rate * 100:.2f}%, Failure Rate: {failure_rate * 100:.2f}")
#plt.xlabel("Sample Number")
#plt.ylabel("Acceleration")
# Create custom legends
#green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Success')
#red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Failure')
#plt.legend(handles=[green_patch, red_patch])
# Add grid
#plt.grid(True)
# Show the plot
#plt.show()

# Scatter plot for Velocity vs Acceleration
#plt.figure(figsize=(8,6))
#for i in range(50):
 #   if grasp_results[i]:
  #      plt.scatter(velocities[i], accelerations[i], color='green', label='Success' if i == 0 else "")
   # else:
    #    plt.scatter(velocities[i], accelerations[i], color='red', label='Failure' if i == 0 else "")

# Add labels and title
#plt.xlabel('Velocity')
#plt.ylabel('Acceleration')
#plt.title(f'Uncertainty in Velocity and Acceleration\n\nSuccess Rate: {success_rate * 100:.2f}%, Failure Rate: {failure_rate * 100:.2f} ')

# Create custom legends
#green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Success')
#red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Failure')
#plt.legend(handles=[green_patch, red_patch])
# Show plot
#plt.show()
