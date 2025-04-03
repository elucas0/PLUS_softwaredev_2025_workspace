# PLUS_softwaredev_2025_workspace
This repository contains the steps on how to perform the assignments from the [*Practice: Software Development Course*](https://github.com/augustinh22/geo-software-dev/tree/main).

## A2 - Recreating Conda Environments
The goal of this task is to go through the process of re-creating and modifying a provided python environment with conda. 

1. **Download the environment files individually:** I manually downloaded the environment files from the course repository https://github.com/augustinh22/geo-software-dev/tree/main/A2.
2. **Creating the first environment** : Using the command `conda env create -f software_dev_v1.yml` I got a wall of errors, likely due to the platform i'm on.
![image](https://github.com/user-attachments/assets/d3a32b85-9101-4609-9f5a-aa8d75fa85de)
4. Looking into the logs, we can see that the command outputs OS dependent-related erros because it uses some windows specific packages.
![image](https://github.com/user-attachments/assets/150a36d4-1287-4cea-8692-1c5c1900dbb7)
5. Using the command `conda env create -f software_dev_v2.yml` I didn't get any error and the packages were installed successfully
![image](https://github.com/user-attachments/assets/4526d087-ec88-4cc8-9582-68bc7c2402e7)
To check that the env was created, I used : `conda activate software_dev_v2` and to check that all the packages were installed : `conda list`
![image](https://github.com/user-attachments/assets/c7100031-abf1-47b7-aa9d-69e3af702ff9)
