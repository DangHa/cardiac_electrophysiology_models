import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create the A matrix in xA = y (with the values are only when i = j and around neighbors)
def set_up_matrix(Mx, My, dx, dy, sigma_x, sigma_y):

    # Define parameters
    M = Mx*My
    rho_x = sigma_x/dx**2
    rho_y = sigma_y/dy**2

    # Define the matrix
    A = np.zeros((M, M))

    for j in range(My):
        for k in range(Mx):

            i = Mx*j + k  # Define global index on x,y axis

            if j == 0 and k == 0:
                A[i, i] = -2*(rho_x + rho_y)
                A[i, i+1] = 2*rho_x
                A[i, i+Mx] = 2*rho_y
            elif j==0 and k == Mx-1:
                A[i, i-1] = 2*rho_x
                A[i, i] = -2*(rho_x + rho_y)
                A[i, i+Mx] = 2*rho_y
            elif j==My-1 and k==0:
                A[i, i-Mx] = 2*rho_y
                A[i, i] = -2*(rho_x + rho_y)
                A[i, i+1] = 2*rho_x
            elif j==My-1 and k==Mx-1:
                A[i, i-Mx] = 2*rho_y
                A[i, i-1] = 2*rho_x
                A[i, i] = -2*(rho_x + rho_y)
            elif k==0:
                A[i, i-Mx] = rho_y
                A[i, i] = -2*(rho_x + rho_y)
                A[i, i+1] = 2*rho_x
                A[i, i+Mx] = rho_y
            elif k==Mx-1:
                A[i, i-Mx] = rho_y
                A[i, i-1] = 2*rho_x
                A[i, i] = -2*(rho_x + rho_y)
                A[i, i+Mx] = rho_y
            elif j==0:
                A[i, i-1] = rho_x
                A[i, i] = -2*(rho_x + rho_y)
                A[i, i+1] = rho_x
                A[i, i+Mx] = 2*rho_y
            elif j==My-1:
                A[i, i-Mx] = 2*rho_y
                A[i, i-1] = rho_x
                A[i, i] = -2*(rho_x + rho_y)
                A[i, i+1] = rho_x
            else:
                A[i, i-Mx] = rho_y
                A[i, i-1] = rho_x
                A[i, i] = -2*(rho_x + rho_y)
                A[i, i+1] = rho_x
                A[i, i+Mx] = rho_y
    
    return A

def show_the_plot(file_path, T, dt, Lx, Ly, dx, dy, save_name):
    # Load the data
    v = pd.read_csv(file_path).to_numpy()
    v_max = np.max(v)
    v_min = np.min(v)

    t = np.arange(0, T+dt, dt)[:, None]        # Time vector
    Mx = round(Lx/dx) + 1                      # Number of point in the x-direction
    My = round(Ly/dy) + 1                      # Number of point in the y-direction

    t_plot = v[:,2::v.shape[1]//16]            # Extract v every 16th column

    # x,y dimension 
    x_axis = np.arange(0, Lx + dx, dx)
    y_axis = np.arange(0, Ly + dy, dy)
    [x, y] = np.meshgrid(x_axis, y_axis)

    f, ax = plt.subplots(4,4, figsize=(16, 16))

    # plots
    ax[0][0].pcolor(x, y, np.reshape(t_plot[:,0], (My, Mx)), vmin=v_min, vmax=v_max)
    ax[0][1].pcolor(x, y, np.reshape(t_plot[:,1], (My, Mx)), vmin=v_min, vmax=v_max)
    ax[0][2].pcolor(x, y, np.reshape(t_plot[:,2], (My, Mx)), vmin=v_min, vmax=v_max)
    ax[0][3].pcolor(x, y, np.reshape(t_plot[:,3], (My, Mx)), vmin=v_min, vmax=v_max)

    ax[1][0].pcolor(x, y, np.reshape(t_plot[:,4], (My, Mx)), vmin=v_min, vmax=v_max)
    ax[1][1].pcolor(x, y, np.reshape(t_plot[:,5], (My, Mx)), vmin=v_min, vmax=v_max)
    ax[1][2].pcolor(x, y, np.reshape(t_plot[:,6], (My, Mx)), vmin=v_min, vmax=v_max)
    ax[1][3].pcolor(x, y, np.reshape(t_plot[:,7], (My, Mx)), vmin=v_min, vmax=v_max)

    ax[2][0].pcolor(x, y, np.reshape(t_plot[:,8], (My, Mx)), vmin=v_min, vmax=v_max)
    ax[2][1].pcolor(x, y, np.reshape(t_plot[:,9], (My, Mx)), vmin=v_min, vmax=v_max)
    ax[2][2].pcolor(x, y, np.reshape(t_plot[:,10], (My, Mx)), vmin=v_max, vmax=v_max)
    ax[2][3].pcolor(x, y, np.reshape(t_plot[:,11], (My, Mx)), vmin=v_min, vmax=v_max)

    ax[3][0].pcolor(x, y, np.reshape(t_plot[:,12], (My, Mx)), vmin=v_min, vmax=v_max)
    ax[3][1].pcolor(x, y, np.reshape(t_plot[:,13], (My, Mx)), vmin=v_min, vmax=v_max)
    ax[3][2].pcolor(x, y, np.reshape(t_plot[:,14], (My, Mx)), vmin=v_min, vmax=v_max)
    bar = ax[3][3].pcolor(x, y, np.reshape(t_plot[:,15], (My, Mx)), vmin=v_min, vmax=v_max)

    # labels
    ax[0][0].title.set_text(f't = {0} ms')
    ax[0][1].title.set_text(f't = {t[len(t)//16*1]} ms')
    ax[0][2].title.set_text(f't = {t[len(t)//16*2]} ms')
    ax[0][3].title.set_text(f't = {t[len(t)//16*3]} ms')

    ax[1][0].title.set_text(f't = {t[len(t)//16*4]} ms')
    ax[1][1].title.set_text(f't = {t[len(t)//16*5]} ms')
    ax[1][2].title.set_text(f't = {t[len(t)//16*6]} ms')
    ax[1][3].title.set_text(f't = {t[len(t)//16*7]} ms')

    ax[2][0].title.set_text(f't = {t[len(t)//16*8]} ms')
    ax[2][1].title.set_text(f't = {t[len(t)//16*9]} ms')
    ax[2][2].title.set_text(f't = {t[len(t)//16*10]} ms')
    ax[2][3].title.set_text(f't = {t[len(t)//16*11]} ms')

    ax[3][0].title.set_text(f't = {t[len(t)//16*12]} ms')
    ax[3][1].title.set_text(f't = {t[len(t)//16*13]} ms')
    ax[3][2].title.set_text(f't = {t[len(t)//16*14]} ms')
    ax[3][3].title.set_text(f't = {t[len(t)//16*15]} ms')

    f.colorbar(bar, ax=ax[3][3], format ='%.2f')
    f.tight_layout()

    plt.savefig(save_name + ".png")
    # plt.show()

def show_AP_at_one_point(file_path, T, dt, Lx, Ly, dx, dy, x, y, save_name):
    Mx = round(Lx/dx) + 1                      # Number of point in the x-direction
    My = round(Ly/dy) + 1                      # Number of point in the y-direction
    t = np.arange(0, T+dt, dt)[:, None]        # Time vector

    v = pd.read_csv(file_path).to_numpy()
    v_reshaped = v.reshape((My, Mx, -1))
    v_point = v_reshaped[int(x/dx), int(y/dy), :]

    f, ax = plt.subplots(1,1, figsize=(6, 6))
    ax.plot(t, v_point)
    ax.title.set_text('v')
    plt.savefig(save_name + ".png")
    # plt.show()
