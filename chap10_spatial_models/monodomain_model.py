import numpy as np
import timeit
import pandas as pd

from helper_functions import set_up_matrix, show_the_plot, show_AP_at_one_point

# Monodomain model with Parsimonious Model for membrane
def monodomain_model(T, dt, Lx, Ly, dx, dy, x_stim, y_stim, save_name):
    start = timeit.default_timer()

    # Set up parameters for bidomain
    Cm = 1           # uF/cm^2
    chi = 2000       # 1/cm
    sigma_i_x = 3.0  # mS/cm
    sigma_i_y = 3.0  # mS/cm
    lamda = 3.0

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
    l_stim = 0.15  # Radius of the stimulation area (in cm)

    # Ionic currents
    I_Na = lambda v, m, h: g_Na*(m**3)*h*(v-v_Na)
    I_K = lambda v: g_K*np.exp(-b_K*(v-v_K))*(v-v_K)
    I_stim = lambda t, x, y: a_stim * (t >= t_stim) * (t <= t_stim + d_stim) * (np.sqrt((x - x_stim)**2 + (y - y_stim)**2) <= l_stim)

    # Define rate constants
    m_inf = lambda v: 1/(1+np.exp((v-Em)/km))
    tau_m = lambda v: 0.12
    h_inf = lambda v: 1/(1+np.exp((v-Eh)/kh))
    tau_h = lambda v: 2*tau_h_0*np.exp(delta_h*(v-Eh)/kh)/(1+np.exp((v-Eh)/kh))

    # Set up discrerization
    N = round(T/dt)                            # Number of time steps
    t = np.arange(0, T+dt, dt)[:, None]        # Time vector
    Mx = round(Lx/dx) + 1                      # Number of point in the x-direction
    My = round(Ly/dy) + 1                      # Number of point in the y-direction
    M = Mx*My                                  # Total number of spatial points

    # Define x and y value on the axises
    x = dx*np.remainder(np.arange(0, M, 1), Mx)
    y = dy*np.floor((np.arange(0, M, 1))/Mx)

    # Set up solution arrays (all physicals points in each time point in time line N)
    v = np.zeros((M, N+1))
    m = np.zeros((M, N+1))
    h = np.zeros((M, N+1))

    # Define initial conditions
    v[:,0] = -83
    m[:,0] = 0
    h[:,0] = 0.9

    # Set up matrices
    Ai = set_up_matrix(Mx, My, dx, dy, sigma_i_x, sigma_i_y)
    I = np.eye(M)
    A = I - ((lamda/((1+lamda)*chi*Cm))*dt)*Ai
    
    # Operator splitting scheme
    for n in range(N):
        # Step 1: Explicit ODE (Parimonious model)
        v[:,n+1] = v[:,n] - (dt/Cm)*(I_Na(v[:,n],m[:,n],h[:,n]) + I_K(v[:,n]) + I_stim(t[n],x,y))
        m[:,n+1] = m[:,n] + dt*((m_inf(v[:,n])-m[:,n])/tau_m(v[:,n]))
        h[:,n+1] = h[:,n] + dt*((h_inf(v[:,n])-h[:,n])/tau_h(v[:,n]))

        # Step 2: Implicit PDE
        v[:,n+1] = np.linalg.solve(A, v[:,n+1] ) # Solve linear system   Ax = b => x = A-1*b

        # Print progress
        if ((n+1) % (N//10) == 0):
            print(f"Running ... {(n+1)//(N//10)*10} %")

    print("Time: " + str(timeit.default_timer() - start) + " s")

    # Save results to CSV
    df = pd.DataFrame(v)
    df.to_csv(save_name, index=False)


if __name__ == "__main__":
    T = 30                                    # Total simulation time (in ms)
    dt = 0.01                                  # Time step (in ms)
    Lx = 1                                     # Length of the domain in the x-direction (in cm)
    Ly = 1                                     # Length of the domain in the y-direction (in cm)
    dx = 0.05                                  # Discretization step in the x-direction (in cm)
    dy = 0.05                                  # Discretization step in the y-direction (in cm)
    x_stim = 0.5                                 # x-coordinate of the point where the AP is to be plotted
    y_stim = 0.5                                 # y-coordinate of the point where the AP is to be plotted
    save_name = "monodomain.csv"                # Name of the file where the membrane potential should be saved

    # Run the monodomain model
    monodomain_model(T, dt, Lx, Ly, dx, dy, x_stim, y_stim, save_name)
    
    # Show the plots
    show_the_plot(save_name, T, dt, Lx, Ly, dx, dy, "monodomain_Vm")

    # Show the AP at one point
    show_AP_at_one_point(save_name, T, dt, Lx, Ly, dx, dy, x_stim, y_stim, "monodomain_AP")
    
    