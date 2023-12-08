import math
import numpy as np
import scipy

from helper_functions import set_up_matrix, show_the_plot

# Parsimonious Model 
def membrane_model():

    # Set up parameters for bidomain
    Cm = 1           # uF/cm^2
    chi = 2000       # 1/cm
    sigma_i_x = 3.0  # mS/cm
    sigma_i_y = 3.0  # mS/cm
    sigma_e_x = 10.0 # mS/cm
    sigma_e_y = 10.0 # mS/cm

    # Set up parameters for membrane model
    g_Na = 11        # mS/cm^2
    g_K = 0.3        # mS/cm^2
    v_Na = 65        # mV
    v_K = -83        # mV
    b_K = 0.047      # 1/mV
    Em, km = -41, -4    
    Eh, kh, tau_h_0, delta_h = -74.9, 4.4, 6.8, 0.8

    # Stimulation parameters
    t_stim = 0     # Stimulation start time (in ms)
    d_stim = 2     # Stimulation current duration (in ms)
    a_stim = -25   # Stimulation current amplitude (in uA/cm^2)
    l_stim = 0.25  # Radius of the stimulation area (in cm)

    # Ionic currents
    I_Na = lambda v, m, h: g_Na*(m**3)*h*(v-v_Na)
    I_K = lambda v: g_K*np.exp(-b_K*(v-v_K))*(v-v_K)
    I_stim = lambda t, x, y: a_stim * (t >= t_stim) * (t <= t_stim + d_stim) * (np.sqrt(x**2 + y**2) <= l_stim)

    # Define rate constants
    m_inf = lambda v: 1/(1+np.exp((v-Em)/km))
    tau_m = lambda v: 0.12
    h_inf = lambda v: 1/(1+np.exp((v-Eh)/kh))
    tau_h = lambda v: 2*tau_h_0*np.exp(delta_h*(v-Eh)/kh)/(1+np.exp((v-Eh)/kh))

    # Set up discrerization
    T = 20                                     # Total simulation time (in ms)
    dt = 0.01                                  # Time step (in ms)
    N = round(T/dt)                            # Number of time steps
    t = np.arange(0, T+dt, dt)[:, None]        # Time vector
    Lx = 1                                     # Length of the domain in the x-direction (in cm)
    Ly = 1                                     # Length of the domain in the y-direction (in cm)
    dx = 0.05                                  # Discretization step in the x-direction (in cm)
    dy = 0.05                                  # Discretization step in the y-direction (in cm)
    Mx = round(Lx/dx) + 1                      # Number of point in the x-direction
    My = round(Ly/dy) + 1                      # Number of point in the y-direction
    M = Mx*My                                  # Total number of spatial points

    # Define x and y value on the axises
    x = dx*np.remainder(np.arange(0, M, 1), Mx)
    y = dy*np.floor((np.arange(0, M, 1))/Mx)

    # Set up solution arrays (all physicals points in each time point in time line N)
    v = np.zeros((M, N+1))
    u = np.zeros((M, N+1))
    m = np.zeros((M, N+1))
    h = np.zeros((M, N+1))

    # Define initial conditions
    v[:,0] = -83
    u[:,0] = 0
    m[:,0] = 0
    h[:,0] = 0.9

    # Set up matrices
    Ai = set_up_matrix(Mx, My, dx, dy, sigma_i_x, sigma_i_y)
    Ae = set_up_matrix(Mx, My, dx, dy, sigma_e_x, sigma_e_y)
    I = np.eye(M)
    A = np.block([[I - (dt/(chi*Cm))*Ai, -(dt/(chi*Cm))*Ai],
                           [Ai, Ai+Ae]])
    b = np.zeros(2*M)
    
    # Adjust the matrix so that ue=0 at the boundary    
    boundary_points = np.concatenate([np.arange(0, Mx), (My-1)*Mx+np.arange(0, Mx), np.arange(1, My-1)*Mx, np.arange(2, My)*Mx-1])
    for boundary_point in boundary_points:
        zero_one_vector = np.zeros(2*M)
        zero_one_vector[M+boundary_point] = 1 # centre area set as 1
        A[M+boundary_point,:] = zero_one_vector

    # Operator splitting scheme
    for n in range(N):
        # Step 1: Explicit ODE (Parimonious model)
        v[:,n+1] = v[:,n] - (dt/Cm)*(I_Na(v[:,n],m[:,n],h[:,n]) + I_K(v[:,n]) + I_stim(t[n],x,y))
        m[:,n+1] = m[:,n] + dt*((m_inf(v[:,n])-m[:,n])/tau_m(v[:,n]))
        h[:,n+1] = h[:,n] + dt*((h_inf(v[:,n])-h[:,n])/tau_h(v[:,n]))

        # Step 2: Implicit PDE
        b[:M] = v[:,n+1]           # Update right-hand side (v^n+0.5)
        vu = np.linalg.solve(A, b) # Solve linear system   Ax = b => x = A-1*b
        v[:,n+1] = vu[:M]          # Extract v
        u[:,n+1] = vu[M:2*M]       # Extract u

        # Print progress
        if ((n+1) % (N//10) == 0):
            print(f"Running ... {(n+1)//(N//10)*10} %")

    # Show the plots
    show_the_plot(v, Mx, My, Lx, Ly, dx, dy)

    show_the_plot(u, Mx, My, Lx, Ly, dx, dy)

# Bidomain model
def bidomain_model():

    return 0


membrane_model()