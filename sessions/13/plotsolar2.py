import sys

from matplotlib.pyplot import plot, show, subplots, title, pause

xlim = 7e11
ylim = 7e11


def plot_bodies(filename):
    fig, ax = subplots()
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            bodies = []
            parts = line.split()
            for i in range(0, len(parts), 4):
                name, x, y, z = parts[i:i + 4]
                x, y = float(x), float(y)
                bodies.append((x, y))
            ax.clear()
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            sizes = [5 if i == 0 else 1 for i in range(len(bodies))]
            ax.scatter([x for x, y in bodies], [y for x, y in bodies], s=sizes)
            pause(1)  # Pause for 1 second to achieve 1 fps
    show()


def simple_plot():
    plot([0, 1, 2], [0, 1, 4])
    title("Simple Plot")
    show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plotsolar2.py <filename>")
        sys.exit(1)
    # simple_plot()
    plot_bodies(sys.argv[1])
