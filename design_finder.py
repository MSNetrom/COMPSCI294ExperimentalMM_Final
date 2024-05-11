import numpy as np
import matplotlib.pyplot as plt


def get_k(alpha, mec, classes):
    return (-1 + np.sqrt(1 + 4 * alpha * mec)) / (2 * alpha)

def get_m(alpha, mec, classes):
    return (-1 + np.sqrt(1 + 4 * alpha * mec)) / 2

def evaluate_capacity(alpha, mec, classes):
    m_vals = get_m(alpha, mec, classes)
    k_vals = get_k(alpha, mec, classes)
    return m_vals, k_vals


if __name__ == "__main__":

    # Get m and k
    classes = 15
    alpha=np.linspace(0.1, 5, 100)
    m_vals, k_vals = evaluate_capacity(alpha=alpha, mec=51.149, classes=classes)

    # Plot
    plt.plot(alpha, m_vals, label="Input-layer")
    plt.plot(alpha, k_vals, label="Hidden-layer")
    plt.plot(alpha, classes*np.ones_like(alpha), label="Out-layer")
    #plt.plot(alpha, np.log2(classes)*np.ones_like(alpha), label=f"Log_2({classes})")
    plt.title("3-layer MLP parametriced by alpha, MEC=51.149 bits")
    plt.xlabel("Alpha")
    plt.ylabel("Layer-nodes")
    plt.legend()
    plt.grid()
    plt.savefig("figures/mlp_capacity_parametriced_by_alpha.pdf", format="pdf")
    plt.show()

    # Get m and k
    alpha = 1
    mec = np.linspace(0, 300, 300)
    m_vals, k_vals = evaluate_capacity(alpha=alpha, mec=mec, classes=classes)

    # Plot
    plt.plot(mec, m_vals, label="Input and hidden layers")
    plt.plot(mec, classes*np.ones_like(mec), label="Out-layer")
    plt.title("3-layer MLP parametriced by MEC, input nodes = hidden nodes")
    plt.xlabel("MEC [bits]")
    plt.ylabel("Layer-nodes")
    plt.legend()
    plt.grid()
    plt.savefig("figures/mlp_capacity_parametriced_by_mec.pdf", format="pdf")
    plt.show()

    # MEC as function of k, and fixed m=5
    m=9
    #k = np.linspace(0, 20, 20)
    #mec = k*(m+1) + k
    mec = np.linspace(0, 160, 160)
    k = mec / (m + 2)
    plt.plot(k, mec, label="MEC from input size")
    plt.title(f"MEC of 3-layer MLP, input nodes = {m}")
    plt.xlabel("Hidden nodes")
    plt.ylabel("MEC [bits]")
    plt.grid()
    plt.savefig("figures/mec_as_function_of_k.pdf", format="pdf")
    plt.show()




    