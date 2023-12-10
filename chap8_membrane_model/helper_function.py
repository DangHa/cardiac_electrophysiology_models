import numpy as np
import matplotlib.pyplot as plt

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
