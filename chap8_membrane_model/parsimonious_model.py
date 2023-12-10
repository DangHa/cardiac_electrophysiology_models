import numpy as np
import timeit
import matplotlib.pyplot as plt

# Parsimonious models
def parsimonious_model():
    # Set up parameters
    Cm = 1        # uF/cm^2
    g_Na = 11      # mS/cm^2
    g_K = 0.3      # mS/cm^2
    v_Na = 65      # mV
    v_K = -83      # mV
    b = 0.047      # 1/mV
    Em, km = -41, -4    
    Eh, kh, tau_h_0, delta_h = -74.9, 4.4, 6.8, 0.8

    # Stimulation parameters
    t_stim = 50
    d_stim = 2
    a_stim = -25

    # Define currents
    I_Na = lambda v, m, h: g_Na*(m**3)*h*(v-v_Na)
    I_K = lambda v: g_K*np.exp(-b*(v-v_K))*(v-v_K)
    I_stim = lambda t: a_stim * (t >= t_stim) * (t <= t_stim + d_stim)

    # Define rate constants
    m_inf = lambda v: 1/(1+np.exp((v-Em)/km))
    tau_m = lambda v: 0.12
    h_inf = lambda v: 1/(1+np.exp((v-Eh)/kh))
    tau_h = lambda v: 2*tau_h_0*np.exp(delta_h*(v-Eh)/kh)/(1+np.exp((v-Eh)/kh))

    # Set up discrerization
    T = 400                                # Total simulation time (in ms)
    dt = 0.001                             # Time step (in ms)
    N = round(T/dt)                        # Number of time steps
    t = np.arange(0, T+dt, dt)[:, None]    # Time vector

    # Set up solution vectors
    v = np.zeros((N+1, 1))
    m = np.zeros((N+1, 1))
    h = np.zeros((N+1, 1))

    # Define initial conditions
    v[0] = -83
    m[0] = 0
    h[0] = 0.9

    # Explicit numerical scheme
    for n in range(N):
        v[n+1] = v[n] - (dt/Cm)*(I_Na(v[n],m[n],h[n]) + I_K(v[n]) + I_stim(t[n]))
        m[n+1] = m[n] + dt*((m_inf(v[n])-m[n])/tau_m(v[n]))
        h[n+1] = h[n] + dt*((h_inf(v[n])-h[n])/tau_h(v[n]))

        # Print progress
        if ((n+1) % (N//10) == 0):
            print(f"Running ... {(n+1)//(N//10)*10} %")

    show_the_plot(t, v, m, h, I_Na(v,m,h), I_K(v), "parsimonious_model")


# show the plots
def show_the_plot(t, v, m, h, I_Na, I_K, save_name):
    f, ax = plt.subplots(2,3, figsize=(12, 6))

    # plots
    ax[0][0].plot(t, v)
    ax[0][1].plot(t, m)
    ax[0][2].plot(t, h)

    ax[1][0].plot(t, I_Na)
    ax[1][1].plot(t[51000:53000], I_Na[51000:53000])
    ax[1][2].plot(t, I_K)
    
    # labels
    ax[0][0].title.set_text('v')
    ax[0][1].title.set_text('m')
    ax[0][2].title.set_text('h')

    ax[1][0].title.set_text('I_Na')
    ax[1][1].title.set_text('I_Na (zoom)')
    ax[1][2].title.set_text('I_K')

    f.tight_layout(pad=2.0)

    plt.savefig(save_name + ".png")
    plt.show()


if __name__ == "__main__":
    start = timeit.default_timer()

    parsimonious_model()

    stop = timeit.default_timer()
    print("Time: " + str(stop - start) + " s")