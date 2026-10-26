import numpy as np
import matplotlib.pyplot as plt
import random

def calculate_tour_cost(tour, distance_matrix):
    cost = 0
    n = len(tour)
    for i in range(n):
        cost += distance_matrix[tour[i]][tour[(i + 1) % n]]
    return cost

def two_opt(tour, distance_matrix):
    best_tour = list(tour)
    improved = True
    improvements_count = 0
    best_cost = calculate_tour_cost(best_tour, distance_matrix)
    
    while improved:
        improved = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                if j - i == 1: continue # Aristas adyacentes no cambian el costo
                new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                new_cost = calculate_tour_cost(new_tour, distance_matrix)
                if new_cost < best_cost:
                    best_tour = new_tour
                    best_cost = new_cost
                    improvements_count += 1
                    improved = True
                    break
            if improved:
                break
    return best_tour, best_cost, improvements_count

def nearest_neighbor(distance_matrix, start_node=None):
    n = len(distance_matrix)
    if start_node is None:
        start_node = random.randint(0, n - 1)
    unvisited = set(range(n))
    unvisited.remove(start_node)
    tour = [start_node]
    
    current = start_node
    while unvisited:
        next_node = min(unvisited, key=lambda city: distance_matrix[current][city])
        unvisited.remove(next_node)
        tour.append(next_node)
        current = next_node
    return tour

if __name__ == '__main__':
    print("--- Ejecutando Ejercicio 3: Búsqueda Local (2-opt TSP) ---")
    
    # Generar instancia de 15 ciudades con coordenadas aleatorias
    num_cities = 15
    np.random.seed(42)
    random.seed(42)
    coords = np.random.rand(num_cities, 2) * 100
    
    # Matriz de distancias euclidianas
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
            
    num_runs = 30
    random_improvements = []
    random_costs = []
    nn_improvements = []
    nn_costs = []
    
    for _ in range(num_runs):
        # 1. Escenario Inicial Aleatorio
        init_random = list(range(num_cities))
        random.shuffle(init_random)
        _, cost_r, imp_r = two_opt(init_random, dist_matrix)
        random_improvements.append(imp_r)
        random_costs.append(cost_r)
        
        # 2. Escenario Inicial con Vecino Más Cercano
        init_nn = nearest_neighbor(dist_matrix)
        _, cost_nn, imp_nn = two_opt(init_nn, dist_matrix)
        nn_improvements.append(imp_nn)
        nn_costs.append(cost_nn)
        
    print(f"Promedio de Mejoras (Aleatorio): {np.mean(random_improvements):.2f} | Costo Final Promedio: {np.mean(random_costs):.2f}")
    print(f"Promedio de Mejoras (Vecino Más Cercano): {np.mean(nn_improvements):.2f} | Costo Final Promedio: {np.mean(nn_costs):.2f}")
    
    # --- Visualización (Boxplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.boxplot([random_improvements, nn_improvements], label=['Aleatorio', 'Vecino Más Cercano'], patch_artist=True)
    ax1.set_title('Número de Mejoras para Converger', fontweight='bold')
    ax1.set_ylabel('Cantidad de Intercambios 2-opt')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.boxplot([random_costs, nn_costs], label=['Aleatorio', 'Vecino Más Cercano'], patch_artist=True)
    ax2.set_title('Calidad de la Solución Final (Costo)', fontweight='bold')
    ax2.set_ylabel('Costo Total del Tour')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle('Evaluación de Búsqueda Local (2-opt) para TSP', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tsp_2opt_comparison.png', dpi=300, bbox_inches='tight')
    print("\n¡Gráfica generada y guardada como 'tsp_2opt_comparison.png'!")
    plt.show()