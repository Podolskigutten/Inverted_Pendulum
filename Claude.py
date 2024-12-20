import pygame
import control as ct
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import cvxopt
from cvxopt import matrix, solvers

@dataclass
class PhysicsParams:
    """Physical parameters of the cart-pole system."""
    gravity: float = 9.81
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_length: float = 1.0
    cart_friction: float = 0.1  # Friction coefficient for cart
    pole_friction: float = 0.01  # Friction coefficient for pole rotation
    max_force: float = 15.0
    min_force: float = -15.0
    
class CartPolePhysics:
    """Handles the physics simulation of the cart-pole system."""
    def __init__(self, params: PhysicsParams):
        self.params = params
        self.total_mass = params.cart_mass + params.pole_mass
        self.pole_mass_length = params.pole_mass * params.pole_length
        
    def equations_of_motion(self, state: np.ndarray, force: float) -> np.ndarray:
        """Calculate state derivatives using equations of motion."""
        x, x_dot, theta, theta_dot = state
        
        # Add friction forces
        cart_friction_force = -self.params.cart_friction * x_dot
        pole_friction_torque = -self.params.pole_friction * theta_dot
        
        total_force = force + cart_friction_force
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        temp = (total_force + self.pole_mass_length * theta_dot**2 * sin_theta) / self.total_mass
        
        # Include pole friction in angular acceleration
        theta_acc = (
            self.params.gravity * sin_theta - 
            cos_theta * temp + 
            pole_friction_torque / (self.params.pole_mass * self.params.pole_length)
        ) / (self.params.pole_length * (4.0/3.0 - self.params.pole_mass * cos_theta**2 / self.total_mass))
        
        x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass
        
        return np.array([x_dot, x_acc, theta_dot, theta_acc])

class Controller(ABC):
    """Abstract base class for controllers."""
    @abstractmethod
    def compute_action(self, state: np.ndarray, **kwargs) -> float:
        pass

class LinearizedModel:
    """Represents a linearized model of the cart-pole system around a specific angle."""
    def __init__(self, nominal_angle: float, physics_params: PhysicsParams):
        self.nominal_angle = nominal_angle
        self.params = physics_params  # Store the PhysicsParams object
        
        # State: [x, x_dot, theta, theta_dot]
        # Compute A and B matrices for linearization around nominal angle
        self.A, self.B = self.linearize()
        
        # Define region of validity (±22.5 degrees from nominal angle)
        self.valid_region = (nominal_angle - np.pi/8, nominal_angle + np.pi/8)
        
    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute linearized A and B matrices."""
        g = self.params.gravity
        M = self.params.cart_mass
        m = self.params.pole_mass
        l = self.params.pole_length
        
        # Nominal state
        theta = self.nominal_angle
        x_dot = 0
        theta_dot = 0
        
        # Partial derivatives for A matrix
        # State vector: [x, x_dot, theta, theta_dot]
        A = np.zeros((4, 4))
        
        # Position derivatives
        A[0, 1] = 1  # dx/dx_dot = 1
        A[2, 3] = 1  # dtheta/dtheta_dot = 1
        
        # Velocity derivatives
        denominator = (M + m*np.sin(theta)**2)
        
        A[1, 2] = -m*l*g*np.cos(theta)/denominator
        A[1, 3] = -m*l*np.sin(theta)/denominator
        
        A[3, 2] = (M + m)*g*np.sin(theta)/(l*denominator)
        A[3, 3] = -m*l*np.cos(theta)*np.sin(theta)/(l*denominator)
        
        # B matrix (control input)
        B = np.zeros((4, 1))
        B[1, 0] = 1/denominator
        B[3, 0] = -np.cos(theta)/(l*denominator)
        
        return A, B

class LinearMPCController:
    """MPC controller for a single linearized model."""
    def __init__(self, model: LinearizedModel, N: int = 10):
        self.model = model
        self.N = N  # Prediction horizon
        
        # Modify cost matrices for better swing-up performance
        self.Q = np.diag([0.1, 0.1, 10.0, 1.0])  # Increase penalty on angle
        self.R = np.array([[0.01]])  # Reduce control cost for more aggressive actions
        
    def setup_qp_matrices(self):
        """Prepare matrices for QP solver."""
        nx = 4  # State dimension
        nu = 1  # Input dimension
        
        # Expand matrices for N steps
        self.Q_bar = np.kron(np.eye(self.N), self.Q)
        self.R_bar = np.kron(np.eye(self.N), self.R)
        
        # Prediction matrices
        A = self.model.A
        B = self.model.B
        
        # Build prediction matrices
        self.Sx = np.zeros((nx * self.N, nx))
        self.Su = np.zeros((nx * self.N, nu * self.N))
        
        temp_A = np.eye(nx)
        for i in range(self.N):
            self.Sx[i*nx:(i+1)*nx, :] = temp_A
            temp_A = A @ temp_A
            
            for j in range(i+1):
                self.Su[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = temp_A @ B
                temp_A = A @ temp_A
        
        # QP matrices
        self.H = 2 * (self.Su.T @ self.Q_bar @ self.Su + self.R_bar)
        self.H = (self.H + self.H.T) / 2  # Ensure symmetry
        
def get_control(self, state: np.ndarray) -> float:
    """Compute optimal control input for current state."""
    # Calculate energy terms
    potential_energy = self.model.params.pole_mass * self.model.params.gravity * \
                      self.model.params.pole_length * (1 - np.cos(state[2]))
    kinetic_energy = 0.5 * self.model.params.pole_mass * \
                     (state[1]**2 + (self.model.params.pole_length * state[3])**2)
    
    # Adjust state relative to linearization point with energy consideration
    delta_state = state - np.array([0, 0, self.model.nominal_angle, 0])
    delta_state[2] += 0.1 * (potential_energy + kinetic_energy)  # Add energy term
    
    # Compute f term in QP
    f = 2 * delta_state.T @ self.Sx.T @ self.Q_bar @ self.Su
    
    # Setup QP constraints
    u_max = self.model.params.max_force
    u_min = self.model.params.min_force
    
    # Input constraints
    G = np.vstack([np.eye(self.N), -np.eye(self.N)])
    h = np.hstack([u_max * np.ones(self.N), -u_min * np.ones(self.N)])
    
    # Solve QP
    try:
        # Convert to cvxopt matrices
        P = matrix(self.H)
        q = matrix(f.T)
        G = matrix(G)
        h = matrix(h)
        
        # Solve
        solvers.options['show_progress'] = False  # Disable solver output
        sol = solvers.qp(P, q, G, h)
        
        if sol['status'] == 'optimal':
            u_sequence = np.array(sol['x'])
            return float(u_sequence[0])  # Return first control input
        else:
            return 0.0  # Fallback control
            
    except:
        return 0.0  # Fallback control

class MultipleLinearMPC(Controller):
    def __init__(self, physics_params: PhysicsParams, N: int = 20):  # Added N parameter
        self.physics_params = physics_params
        self.N = N  # Store N
        
        # Create linearization points every 45 degrees
        angles = np.arange(0, 2*np.pi, np.pi/4)
        
        # Create linear models and controllers
        self.controllers = []
        for angle in angles:
            model = LinearizedModel(angle, self.physics_params)
            controller = LinearMPCController(model, N=self.N)  # Pass N to each controller
            self.controllers.append(controller)
        
        # Add LQR controller for upright stabilization
        self.lqr_controller = self.design_lqr()
        self.lqr_threshold = np.pi/6
    
    def find_best_controller(self, theta: float) -> LinearMPCController:
        """Find the most appropriate controller for current angle."""
        # Normalize angle to [0, 2π)
        theta = theta % (2 * np.pi)
        
        # Find closest linearization point
        distances = [abs(theta - c.model.nominal_angle) for c in self.controllers]
        return self.controllers[np.argmin(distances)]

    def design_lqr(self):
        # Now this will work because we're using the correct physics_params
        upright_model = LinearizedModel(0, self.physics_params)
        
        # LQR weights
        Q = np.diag([1.0, 1.0, 10.0, 1.0])
        R = np.array([[0.1]])
        
        # Compute LQR gains
        K, _, _ = ct.lqr(upright_model.A, upright_model.B, Q, R)
        return K
    
    def compute_action(self, state: np.ndarray, **kwargs) -> float:
        theta = state[2]
        
        # Normalize theta to [-π, π]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # Check if we're near upright position
        if abs(theta) < self.lqr_threshold:
            # Use LQR controller
            return float(-self.lqr_controller @ state)
        else:
            # Use MPC controllers
            controller = self.find_best_controller(theta)
            return controller.get_control(state)

class LQRController(Controller):
    """Linear Quadratic Regulator controller."""
    def __init__(self, physics_params: PhysicsParams):
        # LQR gains calculation (you would typically solve Riccati equation here)
        # State Space Matrices
        self.physics_params = physics_params
        # Parameters
        g = physics_params.gravity
        m_c = physics_params.cart_mass
        m_p = physics_params.pole_mass
        m = m_c + m_p
        l = physics_params.pole_length


        A = [[0, 1, 0, 0],
            [0, 0, (-m_p * g / m) * (1 / (4 / 3 - m_p / m)), 0], 
            [0, 0, 0, 1],
            [0, 0, g / (l * (4 / 3 - m_p / m)), 0]]

        B = [[0],
            [(1 / m) * (1 + m_p * l / (l * (4 / 3 - m_p / m)))],
            [0],
            [(-1 / (l * (4 / 3 - m_p / m))) * 1 / m]]

        C = [[1, 0, 0, 0],
            [0, 0, 1, 0]]

        D = [[0],
            [0]]
        
        # Create a state-space system
        sys = ct.ss(A, B, C, D)

        ## LQR Controller
        Q = np.diag([1,1,1,1])
        R = .1

        self.K, S, E = ct.lqr(sys, Q, R)
        
    def compute_action(self, state: np.ndarray, **kwargs) -> float:
        return float(-self.K @ state)

class KeyboardController(Controller):
    """Manual keyboard control."""
    def __init__(self, force_magnitude: float = 10.0):
        self.force_magnitude = force_magnitude
        
    def compute_action(self, state: np.ndarray, **kwargs) -> float:
        keys = kwargs.get('keys', pygame.key.get_pressed())
        force = 0.0
        if keys[pygame.K_LEFT]:
            force -= self.force_magnitude
        if keys[pygame.K_RIGHT]:
            force += self.force_magnitude
        return force

class CartPoleVisualizer:
    """Handles visualization of the cart-pole system."""
    def __init__(self, width: int = 800, height: int = 400):
        pygame.init()
        self.width = width
        self.height = height
        self.scale = 60  # pixels per meter
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('CartPole Simulation')
        self.clock = pygame.time.Clock()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
    def render(self, state: np.ndarray, force: float) -> None:
        self.screen.fill(self.WHITE)
        
        x, _, theta, _ = state
        cart_x = self.width/2 + x * self.scale
        cart_y = self.height/2
        
        # Draw cart
        cart_width = 60
        cart_height = 30
        pygame.draw.rect(self.screen, self.BLACK,
                        [cart_x - cart_width/2,
                         cart_y - cart_height/2,
                         cart_width,
                         cart_height])
        
        # Draw pole
        pole_x2 = cart_x + self.scale * np.sin(theta)
        pole_y2 = cart_y - self.scale * np.cos(theta)
        pygame.draw.line(self.screen, self.RED,
                        (cart_x, cart_y),
                        (pole_x2, pole_y2),
                        8)
        
        # Draw force arrow
        if abs(force) > 0.1:
            force_x = cart_x - np.sign(force) * min(abs(force), 10.0) * 5
            pygame.draw.line(self.screen, self.BLUE,
                           (cart_x, cart_y),
                           (force_x, cart_y),
                           4)
        
        pygame.display.flip()
        self.clock.tick(50)  # 50 Hz refresh rate
        
    def close(self):
        pygame.quit()

class CartPoleEnv:
    """Main environment class that brings everything together."""
    def __init__(self, 
                 physics_params: Optional[PhysicsParams] = None,
                 controller: Optional[Controller] = None):
        self.physics_params = physics_params or PhysicsParams()
        self.physics = CartPolePhysics(self.physics_params)
        self.controller = controller or KeyboardController()
        self.visualizer = CartPoleVisualizer()
        
        self.dt = 0.02  # 50 Hz simulation
        self.state = self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset the environment state."""
        self.state = (0,0,math.pi*1.01,0)#np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state
        
    def step(self, force: float) -> Tuple[np.ndarray, bool, Dict]:
        """Simulate one timestep."""
        # Clip force to physical limits
        force = np.clip(force, self.physics_params.min_force, self.physics_params.max_force)
        
        # Calculate state derivatives
        derivatives = self.physics.equations_of_motion(self.state, force)
        
        # Euler integration
        self.state = self.state + derivatives * self.dt
        
        # Check termination
        x, _, theta, _ = self.state
        done = bool(abs(x) > 20 )
        
        info = {'force': force}
        return self.state, done, info

def main():
    import matplotlib.pyplot as plt
    
    # Create environment with custom physics parameters
    params = PhysicsParams(
        cart_friction=.5,
        pole_friction=0.01,
        max_force=10.0
    )
    
    # Choose your controller
    #controller = KeyboardController()
    controller =  MultipleLinearMPC(params, N=20)
    #controller = LQRController(params)
    
    env = CartPoleEnv(physics_params=params, controller=controller)
    
    # Lists to store state history
    time_points = []
    x_history = []
    theta_history = []
    force_history = []
    
    done = False
    running = True
    t = 0
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if not done:
                # Get control action
                action = controller.compute_action(
                    env.state,
                    keys=pygame.key.get_pressed()  # Only used by KeyboardController
                )
                
                # Step environment
                state, done, info = env.step(action)
                
                # Store state history
                time_points.append(t)
                x_history.append(state[0])
                theta_history.append(state[2])
                force_history.append(info['force'])
                
                t += env.dt
                
                # Render
                env.visualizer.render(state, info['force'])
            
    finally:
        env.visualizer.close()
        pygame.quit()
        
        # Plot the results
        plt.figure(figsize=(15, 10))
        
        # Plot cart position
        plt.subplot(3, 1, 1)
        plt.plot(time_points, x_history, 'b-', label='Cart Position')
        plt.grid(True)
        plt.ylabel('Position (m)')
        plt.title('Cart Position over Time')
        plt.legend()
        
        # Plot pole angle
        plt.subplot(3, 1, 2)
        plt.plot(time_points, [theta * 180/np.pi for theta in theta_history], 'r-', label='Pole Angle')
        plt.grid(True)
        plt.ylabel('Angle (degrees)')
        plt.title('Pole Angle over Time')
        plt.legend()
        
        # Plot control force
        plt.subplot(3, 1, 3)
        plt.plot(time_points, force_history, 'g-', label='Control Force')
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.title('Control Force over Time')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()