import sys
import matplotlib.pyplot as plt

xlim = 7e11
ylim = 7e11

def update_plot(data, scat):
    parts = data.split()
    if len(parts) == 7:
        name, x, y, z, vx, vy, vz = parts
        x, y = float(x), float(y)
        scat.set_offsets([(x, y)])
        return scat,

def main():
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    scat = ax.scatter([], [])
    trail = True
    while True:
        data = sys.stdin.readline().strip()
        if data == "draw":
        #    if not trail:
        #     ax.cla()
        #     ax.set_xlim(-xlim, xlim)
        #     ax.set_ylim(-ylim, ylim)
            plt.draw()  # Update the plot without pausing
        elif data:
            update_plot(data, scat)

if __name__ == "__main__":
    main()
