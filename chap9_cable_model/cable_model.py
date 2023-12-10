import numpy as np
import timeit
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

# Cable model with Hodgkin Huxley model for membrane
def cable_model():
    # Set up parameters
    Cm = 1          # uF/cm^2
    w = 0.001       # cm
    sigma_i = 4     # mS/cm
    g_Na = 120      # mS/cm^2
    g_K = 36        # mS/cm^2
    g_L = 0.3       # mS/cm^2
    v_Na = 50       # mV
    v_K = -77       # mV
    v_L = -54.4     # mV
    gamma1, gamma2, gamma3 = 0.1, 40, 10  
    gamma4, gamma5, gamma6 = 4, 65, 18    
    gamma7, gamma8, gamma9 = 0.07, 65, 20
    gamma10, gamma11, gamma12 = 1, 35, 10
    gamma13, gamma14, gamma15 = 0.01, 55, 10
    gamma16, gamma17, gamma18 = 0.125, 65, 80

    # Define currents
    I_Na = lambda v, m, h: g_Na*(m**3)*h*(v-v_Na)
    I_K = lambda v, r: g_K*(r**4)*(v-v_K)
    I_L = lambda v: g_L*(v-v_L)
    I_ion = lambda v, m, h, r: I_Na(v, m, h) + I_K(v, r) + I_L(v)

    # Define rate constants
    alpha_m = lambda v: (gamma1*(v+gamma2))/(1-np.exp(-(v+gamma2)/gamma3))
    beta_m = lambda v: gamma4*np.exp(-(v+gamma5)/gamma6)
    alpha_h = lambda v: gamma7*np.exp(-(v+gamma8)/gamma9)
    beta_h = lambda v: gamma10/(1+np.exp(-(v+gamma11)/gamma12))
    alpha_r = lambda v: (gamma13*(v+gamma14))/(1-np.exp(-(v+gamma14)/gamma15))
    beta_r = lambda v: gamma16*np.exp(-(v+gamma17)/gamma18)

    # Set up discrerization
    L = 0.5                              # Length of cell/axon (in cm)
    dx = 0.005                           # Spatial step (in cm)
    M = round(L/dx) + 1                  # Number of spatial points
    T = 5                                # Total simulation time (in ms)
    dt = 0.005                            # Time step (in ms)
    N = round(T/dt)                      # Number of time steps
    t = np.arange(0, T+dt, dt)[:, None]  # Time vector

    # Set up solution vectors
    v = np.zeros((M, N+1))
    m = np.zeros((M, N+1))
    h = np.zeros((M, N+1))
    r = np.zeros((M, N+1))

    # Define initial conditions
    v[:,0] = -65
    v[0:round(0.05/dx),0] = -50
    m[:,0] = 0.1
    h[:,0] = 0.6
    r[:,0] = 0.3

    # Define delta
    delta = w*sigma_i/4

    # Define the matrix
    A = delta/(dx**2)*(spdiags(np.r_[-1, -2*np.ones(M-2), -1], 0, M, M) 
                      + spdiags(np.ones(M), 1, M, M) 
                      + spdiags(np.ones(M), -1, M, M))
    I = np.eye(M)

    # Operator splitting scheme
    for n in range(N):

        # Step 1: Implicit PDE
        v[:,n+1] = np.linalg.solve(I-(dt/Cm)*A, v[:, n])
        m[:,n+1] = m[:, n]
        h[:,n+1] = h[:, n]
        r[:,n+1] = r[:, n]

        # Step 2: Explicit ODE (Hodgkin Huxley model)
        v[:,n+1] = v[:,n+1] - dt/Cm*(I_ion(v[:,n+1],m[:,n+1],h[:,n+1],r[:,n+1]))
        m[:,n+1] = m[:,n+1] + dt*(alpha_m(v[:,n+1])*(1-m[:,n+1]) - beta_m(v[:,n+1])*m[:,n+1])
        h[:,n+1] = h[:,n+1] + dt*(alpha_h(v[:,n+1])*(1-h[:,n+1]) - beta_h(v[:,n+1])*h[:,n+1])
        r[:,n+1] = r[:,n+1] + dt*(alpha_r(v[:,n+1])*(1-r[:,n+1]) - beta_r(v[:,n+1])*r[:,n+1])

        # Print progress
        if ((n+1) % (N//10) == 0):
            print(f"Running ... {(n+1)//(N//10)*10} %")

    show_the_plot(L, dx, v, m, h, r, I_ion(v,m,h,r), "cabel_model")


# show the plots
def show_the_plot(L, dx, v, m, h, r, I_ion, save_name):
    f, ax = plt.subplots(5,4, figsize=(15, 6))

    # x dimension (space)
    x_axis = np.arange(0, L + dx, dx)    

    # get the action potential at time 5 time points [1s, 2s, 3s, 4s]
    v_plot = v[:,::200]
    m_plot = m[:,::200]
    h_plot = h[:,::200]
    r_plot = r[:,::200]
    I_ion_plot = I_ion[:,::200]

    # plots
    ax[0][0].plot(x_axis, v_plot[:,1])
    ax[0][1].plot(x_axis, v_plot[:,2])
    ax[0][2].plot(x_axis, v_plot[:,3])
    ax[0][3].plot(x_axis, v_plot[:,4])
    
    ax[1][0].plot(x_axis, m_plot[:,1])
    ax[1][1].plot(x_axis, m_plot[:,2])
    ax[1][2].plot(x_axis, m_plot[:,3])
    ax[1][3].plot(x_axis, m_plot[:,4])
    
    ax[2][0].plot(x_axis, h_plot[:,1])
    ax[2][1].plot(x_axis, h_plot[:,2])
    ax[2][2].plot(x_axis, h_plot[:,3])
    ax[2][3].plot(x_axis, h_plot[:,4])
    
    ax[3][0].plot(x_axis, r_plot[:,1])
    ax[3][1].plot(x_axis, r_plot[:,2])
    ax[3][2].plot(x_axis, r_plot[:,3])
    ax[3][3].plot(x_axis, r_plot[:,4])

    ax[4][0].plot(x_axis, I_ion_plot[:,1])
    ax[4][1].plot(x_axis, I_ion_plot[:,2])
    ax[4][2].plot(x_axis, I_ion_plot[:,3])
    ax[4][3].plot(x_axis, I_ion_plot[:,4])

    # labels
    cols = ["t = 1 s", "t = 2 s", "t = 3 s", "t = 4 s"]
    rows = ["v (mV)", "m (mV)", "h (mV)", "r (mV)", "I_ion (mV)"]

    for ax_col, col in zip(ax[0], cols):
        ax_col.set_title(col)
        
    for ax_row, row in zip(ax[:,0], rows):
        ax_row.set_ylabel(row, rotation=90, size='large')

    f.tight_layout()

    plt.savefig(save_name + ".png")
    plt.show()


if __name__ == "__main__":
    start = timeit.default_timer()

    cable_model()

    stop = timeit.default_timer()
    print("Time: " + str(stop - start) + " s")