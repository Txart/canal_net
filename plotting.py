import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np

# %%


def plot_water_height(x, Y, bottom):
    plt.figure()
    plt.fill_between(x, y1=bottom, y2=0, color='brown', alpha=0.5)
    plt.fill_between(x, y1=bottom + Y, y2=bottom, color='blue', alpha=0.5)
    plt.title('Initial water height and DEM')


def plot_velocity(x, V):
    plt.figure()
    plt.plot(x, V, color='red')
    plt.title('Initial velocity')


def plot_all_Ys(x, list_of_Ys, bottom, alfa, total_niter_to_plot, n_plots):
    """
    Args:
        x (numpy array): [description]
        list_of_Ys (list(numpy arrays)): [description]
        bottom (numpy array): [description]
        alfa (float): [description]
        total_niter_to_plot (float): higher cut of list_of_Ys.
        n_plots (float): how many plots to show out of all total_niter_to_plot
    """
    plt.figure()
    plt.fill_between(x, y1=bottom, y2=0, color='brown', alpha=0.5)
    plt.title('All Ys')
    for t in range(0, n_plots, int(n_plots/total_niter_to_plot)):
        plt.plot(x, bottom + list_of_Ys[t], alpha=alfa, color='blue')


def plot_all_Qs(x, list_of_Qs, alfa, total_niter_to_plot, n_plots):
    plt.figure()
    plt.title('All Qs')
    for t in range(0, n_plots, int(n_plots/total_niter_to_plot)):
        plt.plot(x, list_of_Qs[t], alpha=alfa, color='red')


def plot_Ys_animation(x, list_of_Ys, bottom, block_height, block_nodes, alfa, total_iterations_to_animate, n_frames, filename):
    figY, axY = plt.subplots()
    axY.set_title("Height of water in canal Y")
    axY.set_ylim(0, np.max(np.array(list_of_Ys)) + max(bottom))
    axY.fill_between(x, y1=bottom, y2=0, color='brown', alpha=0.5)
    # dam
    for i, node in enumerate(block_nodes):
            dam = patches.Rectangle(xy=(
                x[node], bottom[node]), width=0.5, height=block_height[i]-bottom[node], linewidth=1, edgecolor='gray', facecolor='gray')
            axY.add_patch(dam)

    lin, = axY.plot(x, bottom + list_of_Ys[0], alpha=1.0)

    def animate_Y(t):
        lin.set_ydata(bottom + list_of_Ys[t])  # update the data.
        return lin,

    aniY = animation.FuncAnimation(figY, animate_Y, frames=range(
        0, total_iterations_to_animate, int(total_iterations_to_animate/n_frames)))
    aniY.save(filename)


def plot_Qs_animation(x, list_of_Qs, total_iterations_to_animate, n_frames, filename):
    figQ, axQ = plt.subplots()
    axQ.set_title("Animation water flux Q. t = 0")
    axQ.set_ylim(0, np.max(np.array(list_of_Qs)))

    lin, = axQ.plot(x, list_of_Qs[0], alpha=1.0, color='red')

    def animate_Q(t):
        lin.set_ydata(list_of_Qs[t])  # update the data.
        return lin,

    aniQ = animation.FuncAnimation(figQ, animate_Q, frames=range(
        0, total_iterations_to_animate, int(total_iterations_to_animate/n_frames)))
    aniQ.save(filename)


def plot_conservations(list_of_Ys, list_of_Vs, total_iterations):

    plt.figure()
    daily_total_Y = np.sum(np.array(list_of_Ys), axis=1)
    daily_total_V = np.sum(np.array(list_of_Vs), axis=1)
    plt.plot(daily_total_Y, label='Y')
    plt.plot(daily_total_V, label='V')
    plt.title('daily total Y and V')
    plt.xlabel('timesteps')
    plt.ylabel('conserved quantity')
    plt.legend()

# %% Special prints


def print_3_branches_of_Y_shaped_network(qtity, n_nodes):
    nnodes_per_branch = int(n_nodes/3)
    print(f'>>> top branch 1: {qtity[:nnodes_per_branch]}\n')
    print(
        f'>>> top branch 2: {qtity[nnodes_per_branch:2*nnodes_per_branch]}\n')
    print(f'>>> bottom branch: {qtity[2*nnodes_per_branch:]}')
