import numpy as np
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

def show_the_plot(v, Mx, My, Lx, Ly, dx, dy, save_name):
    t_plot = v[:,::500]

    # x,y dimension 
    x_axis = np.arange(0, Lx + dx, dx)
    y_axis = np.arange(0, Ly + dy, dy)
    [x, y] = np.meshgrid(x_axis, y_axis)

    f, ax = plt.subplots(1,4, figsize=(15, 3))

    # plots
    ax[0].pcolor(x, y, np.reshape(t_plot[:,1], (Mx, My)))
    ax[1].pcolor(x, y, np.reshape(t_plot[:,2], (Mx, My)))
    ax[2].pcolor(x, y, np.reshape(t_plot[:,3], (Mx, My)))
    bar = ax[3].pcolor(x, y, np.reshape(t_plot[:,4], (Mx, My)))

    # labels
    ax[0].title.set_text('t = 0.5')
    ax[1].title.set_text('t = 1.0')
    ax[2].title.set_text('t = 1.5')
    ax[3].title.set_text('t = 2.0')

    # show color bar
    f.colorbar(bar, ax=ax[3])
    
    plt.savefig(save_name + ".png")

    plt.show()

