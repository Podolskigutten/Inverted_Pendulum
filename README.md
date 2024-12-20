# Inverted_Pendulum
Energy Controlled Swing-up and LQR Control of Inverted Pendulum on a Cart

CartPole_Simulation.py holds the Cart Pole environments, physics and simulation of the inverted pendulum on a cart.
Some simplification was made, the physics class models the pole similiar to a point mass on a massless rod, while the LQR controlelr linearizes it aorund its operating point theta=0.

Energy controlelr is used for the swing up part and after entering linearized region LQR takes control and stabilizes the pendulum. 

