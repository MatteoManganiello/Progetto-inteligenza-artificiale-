import matplotlib.pyplot as plt

def plot_pil(anni, valori_pil):
    plt.figure(figsize=(10, 6))
    plt.plot(anni, valori_pil, marker='o', linestyle='-', color='b', label='PIL (T)')
    plt.title('Andamento del PIL in Italia dal 2009')
    plt.xlabel('Anno')
    plt.ylabel('PIL')
    plt.grid(True)
    plt.legend()
    plt.show()
