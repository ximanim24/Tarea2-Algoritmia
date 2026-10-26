import numpy as np
import matplotlib.pyplot as plt
import random
import time

def calculate_tour_cost(tour, distance_matrix):
    cost = 0
    n = len(tour)
    for i in range(n):
        cost += distance_matrix[tour[i]][tour[(i + 1) % n]]
    return cost

def get_neighbors_2opt(tour):
    n = len(tour)
    i = random.randint(1, n - 2)
    j = random.randint(i + 1, n - 1)
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

def simulated_annealing(distance_matrix, cooling_schedule='exponential', initial_temp=1000.0, cooling_rate=0.995, max_iter=2000):
    n = len(distance_matrix)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_cost = calculate_tour_cost(current_tour, distance_matrix)
    
    best_tour = list(current_tour)
    best_cost = current_cost
    
    T = initial_temp
    history_costs = [current_cost]
    
    start_time = time.perf_counter()
    
    for iteration in range(max_iter):
        new_tour = get_neighbors_2opt(current_tour)
        new_cost = calculate_tour_cost(new_tour, distance_matrix)
        
        delta = new_cost - current_cost
        
        # Criterio de aceptación de Metropolis
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-8)):
            current_tour = new_tour
            current_cost = new_cost
            if current_cost < best_cost:
                best_tour = list(current_tour)
                best_cost = current_cost
                
        history_costs.append(current_cost)
        
        # Esquema de enfriamiento
        if cooling_schedule == 'exponential':
            T *= cooling_rate
        elif cooling_schedule == 'linear':
            T -= (initial_temp / max_iter)
            if T < 0: T = 0
            
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    return best_tour, best_cost, history_costs, elapsed_time

if __name__ == '__main__':
    print("--- Ejecutando Ejercicio 4: Recocido Simulado (Simulated Annealing) ---")
    
    num_cities = 20
    np.random.seed(42)
    random.seed(42)
    coords = np.random.rand(num_cities, 2) * 100
    
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
            
    # Ejecutar esquema exponencial
    best_t_exp, best_c_exp, history_exp, time_exp = simulated_annealing(
        dist_matrix, cooling_schedule='exponential', initial_temp=1000.0, cooling_rate=0.99, max_iter=3000
    )
    
    # Ejecutar esquema lineal
    best_t_lin, best_c_lin, history_lin, time_lin = simulated_annealing(
        dist_matrix, cooling_schedule='linear', initial_temp=1000.0, max_iter=3000
    )
    
    print(f"Exponencial -> Costo Final: {best_c_exp:.2f} | Tiempo: {time_exp:.4f}s")
    print(f"Lineal      -> Costo Final: {best_c_lin:.2f} | Tiempo: {time_lin:.4f}s")
    
    # --- Visualización 1: Curvas de Convergencia ---
    plt.figure(figsize=(9, 5))
    plt.plot(history_exp, label='Enfriamiento Exponencial', color='#1f77b4', linewidth=1.5)
    plt.plot(history_lin, label='Enfriamiento Lineal', color='#ff7f0e', linewidth=1.5, linestyle='--')
    plt.title('Curvas de Convergencia: Simulated Annealing (Costo vs. Iteraciones)', fontweight='bold')
    plt.xlabel('Iteraciones', fontsize=11)
    plt.ylabel('Costo del Tour', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('sa_convergence.png', dpi=300, bbox_inches='tight')
    print("¡Gráfica 1 guardada como 'sa_convergence.png'!")
    plt.show()
    
    # --- Visualización 2: Tiempo vs Calidad ---
    fig, ax = plt.subplots(figsize=(7, 5))
    schedules = ['Exponencial', 'Lineal']
    costs = [best_c_exp, best_c_lin]
    times = [time_exp, time_lin]
    
    scatter = ax.scatter(times, costs, c=['#1f77b4', '#ff7f0e'], s=150, zorder=3)
    for i, txt in enumerate(schedules):
        ax.annotate(txt, (times[i], costs[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
        
    ax.set_title('Tiempo de Ejecución vs. Calidad de la Solución Final', fontweight='bold')
    ax.set_xlabel('Tiempo de Ejecución (segundos)', fontsize=11)
    ax.set_ylabel('Costo Final del Tour', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('sa_time_vs_quality.png', dpi=300, bbox_inches='tight')
    print("¡Gráfica 2 guardada como 'sa_time_vs_quality.png'!")
    plt.show()