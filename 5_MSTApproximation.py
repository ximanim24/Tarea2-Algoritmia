import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

def calculate_tour_cost(tour, distance_matrix):
    cost = 0
    n = len(tour)
    for i in range(n):
        cost += distance_matrix[tour[i]][tour[(i + 1) % n]]
    return cost

def get_exact_tsp_opt(distance_matrix):
    n = len(distance_matrix)
    best_cost = float('inf')
    # Fijar el nodo 0 para evitar permutaciones cíclicas repetidas
    nodes = list(range(1, n))
    for perm in itertools.permutations(nodes):
        tour = [0] + list(perm)
        cost = calculate_tour_cost(tour, distance_matrix)
        if cost < best_cost:
            best_cost = cost
    return best_cost

def prim_mst(distance_matrix):
    n = len(distance_matrix)
    selected = [False] * n
    parent = [-1] * n
    key = [float('inf')] * n
    
    key[0] = 0
    parent[0] = -1
    
    for _ in range(n):
        u = -1
        min_val = float('inf')
        for v in range(n):
            if not selected[v] and key[v] < min_val:
                min_val = key[v]
                u = v
        selected[u] = True
        
        for v in range(n):
            if distance_matrix[u][v] < key[v] and not selected[v]:
                key[v] = distance_matrix[u][v]
                parent[v] = u
                
    # Construir lista de adyacencia del MST
    adj = {i: [] for i in range(n)}
    for i in range(1, n):
        adj[parent[i]].append(i)
        adj[i].append(parent[i])
    return adj

def dfs_preorder(adj, root=0):
    visited = set()
    tour = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            tour.append(node)
            # Agregar vecinos en orden inverso para recorrido preorden correcto
            for neighbor in reversed(adj[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    return tour

def two_opt_single(tour, distance_matrix):
    best_tour = list(tour)
    improved = True
    best_cost = calculate_tour_cost(best_tour, distance_matrix)
    while improved:
        improved = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                if j - i == 1: continue
                new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                new_cost = calculate_tour_cost(new_tour, distance_matrix)
                if new_cost < best_cost:
                    best_tour = new_tour
                    best_cost = new_cost
                    improved = True
                    break
            if improved: break
    return best_tour, best_cost

if __name__ == '__main__':
    print("--- Ejecutando Ejercicio 5: Razón de Aproximación (MST Approximation) ---")
    
    num_instances = 100
    num_cities = 9  # 9 ciudades permite calcular el óptimo exacto por fuerza bruta rápidamente (9! / 2)
    
    ratios_mst = []
    costs_mst_pure = []
    costs_mst_2opt = []
    costs_opt = []
    costs_sa = []
    
    np.random.seed(42)
    random.seed(42)
    
    for _ in range(num_instances):
        coords = np.random.rand(num_cities, 2) * 100
        dist_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
                
        # 1. Óptimo Exacto
        c_opt = get_exact_tsp_opt(dist_matrix)
        costs_opt.append(c_opt)
        
        # 2. Algoritmo de Aproximación por MST (Preorden)
        mst_adj = prim_mst(dist_matrix)
        mst_tour = dfs_preorder(mst_adj, root=0)
        c_mst = calculate_tour_cost(mst_tour, dist_matrix)
        costs_mst_pure.append(c_mst)
        
        # Razón de aproximación (C_obtenido / C_optimo)
        ratios_mst.append(c_mst / c_opt)
        
        # 3. MST + 2-opt como Post-procesamiento
        _, c_mst_2opt = two_opt_single(mst_tour, dist_matrix)
        costs_mst_2opt.append(c_mst_2opt)
        
        # Simulación rápida para comparar con Ejercicio 4
        # (Usamos una aproximación del costo de SA basado en 2-opt aleatorio)
        init_rand = list(range(num_cities))
        random.shuffle(init_rand)
        _, c_sa = two_opt_single(init_rand, dist_matrix)
        costs_sa.append(c_sa)
        
    print(f"Razón de Aproximación Promedio (MST): {np.mean(ratios_mst):.4f} (Lote máximo teórico: 2.0)")
    
    # --- Visualización 1: Histograma de Razón de Aproximación ---
    plt.figure(figsize=(8, 5))
    plt.hist(ratios_mst, bins=15, color='#2ca02c', edgecolor='black', alpha=0.7)
    plt.axvline(2.0, color='#d62728', linestyle='--', linewidth=2, label='Límite Teórico (2.0)')
    plt.axvline(np.mean(ratios_mst), color='#1f77b4', linestyle='-', linewidth=2, label=f'Promedio Observado ({np.mean(ratios_mst):.2f})')
    plt.title('Histograma de la Razón de Aproximación ($C_{\text{MST}} / C_{\text{óptimo}}$)', fontweight='bold')
    plt.xlabel('Razón de Aproximación', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('mst_ratio_histogram.png', dpi=300, bbox_inches='tight')
    print("¡Gráfica 1 guardada como 'mst_ratio_histogram.png'!")
    plt.show()
    
    # --- Visualización 2: Comparativa de Costos entre Métodos ---
    plt.figure(figsize=(9, 5))
    data_to_plot = [costs_opt, costs_mst_pure, costs_mst_2opt, costs_sa]
    label_plot = ['Óptimo Exacto', 'MST Puro', 'MST + 2-opt', 'Búsqueda Local']
    
    box = plt.boxplot(data_to_plot, label=label_plot, patch_artist=True)
    colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    plt.title('Comparativa de Costos del Tour entre Métodos (100 instancias)', fontweight='bold')
    plt.ylabel('Costo Total del Tour', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('methods_cost_comparison.png', dpi=300, bbox_inches='tight')
    print("¡Gráfica 2 guardada como 'methods_cost_comparison.png'!")
    plt.show()