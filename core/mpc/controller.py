import numpy as np
from scipy.optimize import minimize
import logging

class MPCController:
    """
    Model Predictive Control for Mobile VLA Navigation.
    Optimizes a 2-DOF control sequence (linear_x, linear_y) over a receding horizon.
    """
    
    def __init__(self, horizon=10, dt=0.2, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.N = horizon      # Look-ahead steps
        self.dt = dt          # Time step duration
        
        # Constraints
        self.max_v = kwargs.get('max_v', 1.15)        # m/s
        self.d_safe = kwargs.get('d_safe', 0.5)      # Safety margin in meters
        
        # Weights for cost function
        self.w_goal = kwargs.get('w_goal', 1.0)
        self.w_obs = kwargs.get('w_obs', 5.0)
        self.w_smooth = kwargs.get('w_smooth', 0.1)
        self.w_ctrl = kwargs.get('w_ctrl', 0.05)
        
    def dynamics(self, state, u):
        """
        Calculates next state based on current state and control input.
        state: [x, y, theta]
        u: [vx, vy]
        """
        x, y, theta = state
        vx, vy = u
        
        # Simple integrator model for 2-DOF navigation
        # (Assuming holonomic-like movement for linear_x, linear_y in robot frame)
        new_x = x + vx * self.dt
        new_y = y + vy * self.dt
        new_theta = theta # Assuming 2nd line of VLA action handles rotation implicitly or we focus on translation
        
        return np.array([new_x, new_y, new_theta])

    def cost_function(self, u_flat, current_state, goal, obstacles):
        """
        Total cost = Goal + Obstacles + Smoothness + Control efforts.
        """
        u = u_flat.reshape(self.N, 2)
        total_cost = 0.0
        state = current_state.copy()
        
        for k in range(self.N):
            # 1. Update state
            state = self.dynamics(state, u[k])
            
            # 2. Goal Cost (Distance to target)
            dist_to_goal = np.linalg.norm(state[:2] - goal[:2])
            # Weight goal more towards the end of horizon? (Optional)
            total_cost += self.w_goal * (dist_to_goal**2)
            
            # 3. Obstacle Cost (Barrier function)
            for obs in obstacles:
                # obs format: {"x": x, "y": y, "r": radius}
                dist_to_obs = np.linalg.norm(state[:2] - np.array([obs['x'], obs['y']]))
                if dist_to_obs < self.d_safe:
                    # Exponential or quadratic penalty for violating safety margin
                    total_cost += self.w_obs * (self.d_safe - dist_to_obs)**2 * 10.0
            
            # 4. Control Effort Cost
            total_cost += self.w_ctrl * np.linalg.norm(u[k])**2
            
            # 5. Smoothness Cost (Difference from previous control)
            if k > 0:
                total_cost += self.w_smooth * np.linalg.norm(u[k] - u[k-1])**2
                
        return total_cost

    def solve(self, current_state, goal, obstacles):
        """
        Finds the optimal control sequence.
        Returns: The first control input [vx, vy] (Receding Horizon).
        """
        # Initial guess: zero controls
        u0 = np.zeros(self.N * 2)
        
        # Bounds: [vx_min, vx_max], [vy_min, vy_max]
        bounds = [(-self.max_v, self.max_v)] * (self.N * 2)
        
        # Optimization
        result = minimize(
            self.cost_function,
            u0,
            args=(current_state, goal, obstacles),
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-3, 'disp': False}
        )
        
        if not result.success:
            self.logger.warning(f"MPC optimization failed: {result.message}")
            
        # Reshape and take the first action
        u_opt = result.x.reshape(self.N, 2)
        return u_opt[0], u_opt

if __name__ == "__main__":
    # Test MPC Solver
    mpc = MPCController(horizon=5)
    
    start_state = np.array([0.0, 0.0, 0.0])
    target_goal = np.array([2.0, 0.0, 0.0])
    obs_list = [{"x": 1.0, "y": 0.1, "r": 0.3}]
    
    action, trajectory = mpc.solve(start_state, target_goal, obs_list)
    
    print("Initial Action:", action)
    print("Trajectory:\n", trajectory)
