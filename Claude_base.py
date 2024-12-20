import pygame
import matplotlib.pyplot as plt
import control as ct
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import cvxpy as cp
from scipy import sparse


@dataclass
class PhysicsParams:
    """Physical parameters of the cart-pole system."""
    gravity: float = 9.81
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_length: float = 1.0
    cart_friction: float = 0.5  # Friction coefficient for cart
    pole_friction: float = .05  # Friction coefficient for pole rotation
    max_force: float = 50.0
    min_force: float = -50.0

class Controller(ABC):
    """Abstract base class for controllers."""
    @abstractmethod
    def compute_action(self, state: np.ndarray, **kwargs) -> float:
        pass

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
        
        #theta_acc = (self.params.gravity * sin_theta + cos_theta*(-total_force - self.params.pole_mass*theta_dot**2*sin_theta))/(self.params.pole_length*(4/3 - self.params.pole_mass * cos_theta**2 / self.total_mass))
        
        #x_acc = (total_force + self.params.pole_length*self.params.pole_mass*theta_dot**2*sin_theta - self.params.pole_length*self.params.pole_mass*theta_acc*cos_theta)/(self.params.cart_mass + self.params.pole_mass)
        return np.array([x_dot, x_acc, theta_dot, theta_acc])

class CartPoleVisualizer:
    """Handles visualization of the cart-pole system."""
    def __init__(self, width: int = 1200, height: int = 600):
        pygame.init()
        self.width = width
        self.height = height
        self.scale = 100  # pixels per meter
        
        # Define physical dimensions in meters
        self.cart_width_meters = 0.5  # 50cm wide cart
        self.cart_height_meters = 0.25  # 25cm tall cart
        self.pole_length_meters = 1.0  # 1m pole
        self.pole_width_meters = 0.05  # 5cm thick pole
        
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
        
        # Draw cart - scale physical dimensions
        cart_width = self.cart_width_meters * self.scale
        cart_height = self.cart_height_meters * self.scale
        pygame.draw.rect(self.screen, self.BLACK,
                        [cart_x - cart_width/2,
                         cart_y - cart_height/2,
                         cart_width,
                         cart_height])
        
        # Draw pole - scale pole length and width
        pole_length = self.pole_length_meters * self.scale
        pole_width = self.pole_width_meters * self.scale
        pole_x2 = cart_x + pole_length * np.sin(theta)
        pole_y2 = cart_y - pole_length * np.cos(theta)
        pygame.draw.line(self.screen, self.RED,
                        (cart_x, cart_y),
                        (pole_x2, pole_y2),
                        int(pole_width))  # Convert width to integer for pygame
        
        # Draw force arrow - scale with system
        if abs(force) > 0.1:
            force_scale = self.scale * 0.1  # Scale force arrow with rest of system
            force_x = cart_x - np.sign(force) * min(abs(force), 20.0) * force_scale
            pygame.draw.line(self.screen, self.BLUE,
                           (cart_x, cart_y),
                           (force_x, cart_y),
                           int(pole_width/2))  # Make force arrow width proportional to pole
        
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
        x = 0
        x_dot = 0
        theta = 180
        theta_dot = 0

        self.state = (x,x_dot,-math.radians(theta),theta_dot)#np.random.uniform(low=-0.05, high=0.05, size=(4,))
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
        
        ## LQR Controller
        Q = np.diag([1,1,10,1])
        R = .1

        self.K, S, E = ct.lqr(A, B, Q, R)
                
    def compute_action(self, state: np.ndarray, **kwargs) -> float:
        return float(-self.K @ state)

class KeyboardController(Controller):
    """Manual keyboard control."""
    def __init__(self, force_magnitude: float = 20.0):
        self.force_magnitude = force_magnitude
        
    def compute_action(self, state: np.ndarray, **kwargs) -> float:
        keys = kwargs.get('keys', pygame.key.get_pressed())
        force = 0.0
        if keys[pygame.K_LEFT]:
            force -= self.force_magnitude
        if keys[pygame.K_RIGHT]:
            force += self.force_magnitude
        return force

class EnergyController(Controller):
    def __init__(self, physics_params: PhysicsParams, k_energy: float = 13.0, k_pos: float = 1.0):
        self.params = physics_params
        self.k_energy = k_energy  
        self.k_pos = k_pos      
        
    def compute_action(self, state: np.ndarray, **kwargs) -> float:
        x, x_dot, theta, theta_dot = state
        
        # Normalize angle to [-π, π] or something
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        
        # Simple swing logic:
        # If pendulum is near bottom (close to ±π) 
        # Push in direction of motion to amplify swing
        if abs(abs(theta) - np.pi) < 0.5:  # Near bottom
            force = self.k_energy * np.sign(theta_dot)
        else:
            force = 0
            
        # Basic position control to keep cart centered
        force += -self.k_pos * x - 0.5 * x_dot
        
        return float(np.clip(force, self.params.min_force, self.params.max_force))

    def compute_energy(self, state: np.ndarray) -> float:
        _, x_dot, theta, theta_dot = state
        
        potential_energy = self.params.pole_mass * self.params.gravity * \
                         self.params.pole_length * (1 - np.cos(theta))
        
        kinetic_energy_pole = 0.5 * self.params.pole_mass * \
                        (self.params.pole_length * theta_dot)**2
        
        kinetic_energy_cart = 0.5 * self.params.cart_mass * x_dot**2
        
        return potential_energy + kinetic_energy_pole + kinetic_energy_cart
    

class HybridController(Controller):
    def __init__(self, physics_params):
        self.params = physics_params
        self.energy_controller = EnergyController(physics_params)
        self.lqr_controller = LQRController(physics_params)
        self.ANGLE_TO_LQR = math.radians(10)  # Stricter angle threshold
        self.ANGLE_TO_SWINGUP = math.radians(45)  # Wider threshold to prevent quick switches
        self.VELOCITY_THRESHOLD = 2.0  # Add velocity threshold
        self.current_mode = "swingup"
        
    def compute_action(self, state, **kwargs):
        x, x_dot, theta, theta_dot = state
        
        # Normalize angle consistently
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        
        # Switch to LQR only when both angle AND velocity are small enough
        if self.current_mode == "swingup":
            if (abs(theta) < self.ANGLE_TO_LQR and 
                abs(theta_dot) < self.VELOCITY_THRESHOLD):
                self.current_mode = "lqr"
                # Add a small delay or ramp-up period for LQR
                return 0.0  # Zero force for one timestep during transition
        
        # Switch back to swing-up if angle gets too large
        elif self.current_mode == "lqr":
            if abs(theta) > self.ANGLE_TO_SWINGUP:
                self.current_mode = "swingup"
        
        # Prepare state for controllers
        if self.current_mode == "lqr":
            # For LQR, convert to equivalent angle near 0
            if abs(abs(theta) - np.pi) < 0.5:
                theta = -np.sign(theta) * (2 * np.pi - abs(theta))
        
        state = np.array([x, x_dot, theta, theta_dot])
        
        # Apply force limit during transitions
        force = 0.0
        if self.current_mode == "swingup":
            force = self.energy_controller.compute_action(state, **kwargs)
        else:
            raw_force = self.lqr_controller.compute_action(state, **kwargs)
            # Limit initial LQR force to prevent spikes
            force = np.clip(raw_force, -5.0, 5.0)  # Start with smaller force limits
            
        return force


def main():
    # Create environment with custom physics parameters
    params = PhysicsParams(
        cart_friction=.5,
        pole_friction=0.05,
    )
    
    # Create hybrid controller
    controller = HybridController(params)
    env = CartPoleEnv(physics_params=params, controller=controller)
    
    # Lists to store state history
    time_points = []
    x_history = []
    theta_history = []
    force_history = []
    #controller_phase = []  # Track which controller is active
    
    done = False
    running = True
    t = 0
    state = env.reset()
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if not done:
                # Get control action from hybrid controller
                action = controller.compute_action(
                    env.state,
                    keys=pygame.key.get_pressed()
                )
                
                # Step environment
                state, done, info = env.step(action)
                
                # Store state history
                time_points.append(t)
                x_history.append(state[0])
                theta_history.append(state[2])
                force_history.append(info['force'])
                #controller_phase.append(controller.current_mode)
                
                t += env.dt
                
                # Render
                env.visualizer.render(state, info['force'])
            
    finally:
        env.visualizer.close()
        pygame.quit()
        
        # Plot the results
        plt.figure(figsize=(15, 12))
        
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
        plt.ylabel('Force (N)')
        plt.title('Control Force over Time')
        plt.legend()
        
        # Plot controller phase
        """plt.subplot(2, 2, 4)
        phase_numeric = [1 if p == "lqr" else 0 for p in controller_phase]
        plt.plot(time_points, phase_numeric, 'k-', label='Controller Phase')
        plt.grid(True)
        plt.ylabel('Phase')
        plt.xlabel('Time (s)')
        plt.yticks([0, 1], ['Swing-up', 'LQR'])
        plt.title('Active Controller over Time')
        plt.legend()"""
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()