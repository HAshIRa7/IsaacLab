#### Picking Evaluaion

![](https://github.com/HAshIRa7/IsaacLab/blob/feat/grasps/picking.gif)  


### SETUP 

IsaacSim - 4.5.0, IsaacLab - v2.0.2

Robot - Franka-Panda 

Observations - joint positions, joint velocities, object position in gripper frame, one hot encoding of object types

Policy - MLP [256, 128, 64] 

Actions - Franka joint positions + binary gripper close/open

Training on Laptop 4080 - 512 envs with NO Cameras, 

Also setuping samera with depth and RGB lead to bad perfomance - only 50 envs available + iteration from 1s to 6s 

Grasps - pose for tetra pack only orthogonal for origin of pack, can-tin can pick for both sides  

For experiment evaluation see ![analys](https://github.com/HAshIRa7/IsaacLab/blob/feat/grasps/analysis.ipynb)   

##### Training process 

![](https://github.com/HAshIRa7/IsaacLab/blob/feat/grasps/train.gif)    

#### Can create different envs with different object orientation 

![](https://github.com/HAshIRa7/IsaacLab/blob/feat/grasps/another_train_setup.gif)
![](https://github.com/HAshIRa7/IsaacLab/blob/feat/grasps/second_training_sample.gif)

##### Usd files  

Create URDF and use this script [link](https://github.com/YitianShi/MetaIsaacGrasp/blob/main/urdf_converter.py)