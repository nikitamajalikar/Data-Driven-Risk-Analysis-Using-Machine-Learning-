**Robotic Manipulation Under Uncertainty**

This project explores the impact of uncertainty in robotic manipulation by modeling variations in object position, gripper width, velocity, and acceleration. Using the Franka Emika Panda collaborative robot, combined with ROS, Gazebo simulation, MoveIt motion planning, and OpenCV vision, the framework evaluates grasp success rates in both simulated and real-world environments.

The study demonstrates how different probabilistic distributions (Normal, Uniform) influence robotic pick-and-place performance and highlights strategies to improve system robustness in uncertain industrial scenarios.

**Key Features**

Uncertainty Modeling → Simulates variability in object position, gripper width, and motion parameters using probabilistic distributions.

Simulation with Gazebo → Safe and realistic testing of grasp scenarios before real-world deployment.

ROS + MoveIt Integration → Motion planning, trajectory optimization, and grasp execution under uncertainty.

Real-World Validation → Experiments with the Franka Emika Panda robot to confirm simulation findings.

Data Collection & Analysis → Automated logging of success/failure rates, positional errors, and performance metrics.

Practical Insights → Identifies critical factors (e.g., object position, gripper width) that most affect grasp reliability.
