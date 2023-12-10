import numpy as np
import timeit
import matplotlib.pyplot as plt

# Hodgkin Huxley Model
def hodgkin_huxley_model():
    # Set up parameters
    Cm = 1          # uF/cm^2
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

    # Define rate constants
    alpha_m = lambda v: (gamma1*(v+gamma2))/(1-np.exp(-(v+gamma2)/gamma3))
    beta_m = lambda v: gamma4*np.exp(-(v+gamma5)/gamma6)
    alpha_h = lambda v: gamma7*np.exp(-(v+gamma8)/gamma9)
    beta_h = lambda v: gamma10/(1+np.exp(-(v+gamma11)/gamma12))
    alpha_r = lambda v: (gamma13*(v+gamma14))/(1-np.exp(-(v+gamma14)/gamma15))
    beta_r = lambda v: gamma16*np.exp(-(v+gamma17)/gamma18)

    # Set up discrerization
    T = 10                                 # Total simulation time (in ms)
    dt = 0.001                             # Time step (in ms)
    N = round(T/dt)                        # Number of time steps
    t = np.arange(0, T+dt, dt)[:, None]    # Time vector

    # Set up solution vectors
    v = np.zeros((N+1, 1))
    m = np.zeros((N+1, 1))
    h = np.zeros((N+1, 1))
    r = np.zeros((N+1, 1))

    # Define initial conditions
    v[0] = -60
    m[0] = 0.1
    h[0] = 0.6
    r[0] = 0.3

    # Explicit numerical scheme
    for n in range(N):
        v[n+1] = v[n] - (dt/Cm)*(I_Na(v[n],m[n],h[n]) + I_K(v[n],r[n]) + I_L(v[n]))
        m[n+1] = m[n] + dt*(alpha_m(v[n])*(1-m[n]) - beta_m(v[n])*m[n])
        h[n+1] = h[n] + dt*(alpha_h(v[n])*(1-h[n]) - beta_h(v[n])*h[n])
        r[n+1] = r[n] + dt*(alpha_r(v[n])*(1-r[n]) - beta_r(v[n])*r[n])

        # Print progress
        if ((n+1) % (N//10) == 0):
            print(f"Running ... {(n+1)//(N//10)*10} %")

    show_the_plot(t, v, m, h, r, I_Na(v,m,h), I_K(v,r), I_L(v), "hodgkin_huxley_model")


# show the plots
def show_the_plot(t, v, m, h, r, I_Na, I_K, I_L, save_name):
    f, ax = plt.subplots(2,4, figsize=(15, 6))

    # plots
    ax[0][0].plot(t, v)
    ax[0][1].plot(t, m)
    ax[0][2].plot(t, h)
    ax[0][3].plot(t, r)

    ax[1][0].plot(t, I_Na)
    ax[1][1].plot(t, I_K)
    ax[1][2].plot(t, I_L)
    
    # labels
    ax[0][0].title.set_text('v')
    ax[0][1].title.set_text('m')
    ax[0][2].title.set_text('h')
    ax[0][3].title.set_text('r')

    ax[1][0].title.set_text('I_Na')
    ax[1][1].title.set_text('I_K')
    ax[1][2].title.set_text('I_L')

    f.delaxes(ax[1][3])
    f.tight_layout(pad=2.0)

    plt.savefig(save_name + ".png")
    plt.show()


if __name__ == "__main__":
    start = timeit.default_timer()

    hodgkin_huxley_model()

    stop = timeit.default_timer()
    print("Time: " + str(stop - start) + " s")