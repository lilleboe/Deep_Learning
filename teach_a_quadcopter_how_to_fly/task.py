import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 10
        self.action_high = 900
        self.action_size = 4
        self.runtime = runtime

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        #self.init_pos = self.sim
        self.target_z = self.target_pos[2]
        self.target_dist = float(abs(self.target_z - self.sim.pose[2]))
        #print(self.target_dist)

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        current_z = self.sim.pose[2]
        v_velocity = self.sim.v[2]

        #reward = np.clip(1. + (current_z - self.target_z) / max(self.target_dist, 1.), 0., 1.)
        #reward = np.clip(reward + np.clip(v_velocity, -.2, .2), 0., 1.)
  
        reward = np.clip(v_velocity, -0.5, 0.5)
        if current_z >= self.target_z:
            reward = 1.0

        # Penalize if we are "done" and have not exceeded the time
        # and we are still below the target height
        if done and self.sim.time < self.sim.runtime and self.target_z > current_z:
            reward = -1.0
        return reward
        
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        
        # cut short the steps if we have reached the desired z height
        #current_z = self.sim.pose[2]
        #if self.target_z <= current_z:
        #    done = True
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state