#### Tetra Pack picking

![](https://github.com/HAshIRa7/IsaacLab/blob/feat/grasps/tetra_pack.gif) 

#### Conserva picking

![](https://github.com/HAshIRa7/IsaacLab/blob/feat/grasps/cons.gif)  



### SETUP 


Robot - Franka-Panda 

Observations - joint positions, joint velocities, object position in gripper frame, one hot encoding of object types

Policy - MLP [256, 128, 64]

Training on Laptop 4080 - 512 envs with NO Cameras, 

Also setuping samera with depth and RGB lead to bad perfomance - only 50 envs available + iteration from 1s to 6s 

Success rate: tetra-pack: ~ 78%, tin-can - 80%, chips-bag - 0%. Can't pick up chips even if training with only it. 
Grasps - pose for tetra pack only orthogonal for origin of pack, can-tin can pick for both sides 



##### Training process 

![](https://github.com/HAshIRa7/IsaacLab/blob/feat/grasps/train.gif)  