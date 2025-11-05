import random
import math
from scipy.stats import bernoulli

# ============================================================================
# FUNCIÓN OBJETIVO: OneMax
# ============================================================================
def onemax(individual):
    """
    Función OneMax: suma de todos los bits en el individuo.
    Objetivo: maximizar (encontrar vector de todos unos).
    
    Args:
        individual: lista de 0s y 1s
    Returns:
        suma de elementos (fitness)
    """
    fitness = 0
    for i in range(len(individual)):
        fitness += individual[i]
    return fitness


# ============================================================================
# UMDA DISCRETO - ALGORITMO PRINCIPAL
# ============================================================================
def umda_discreto(n, population_size, elite_percentage, max_generations, seed=None):
    """
    Implementación del Univariate Marginal Distribution Algorithm (UMDA).
    
    Args:
        n: dimensión del problema (longitud del vector binario)
        population_size: tamaño de la población
        elite_percentage: porcentaje de elite (e.g., 0.4 para 40%)
        max_generations: número máximo de generaciones
        seed: semilla para reproducibilidad
        
    Returns:
        diccionario con resultados del algoritmo
    """
    if seed is not None:
        random.seed(seed)
    
    # Calcular tamaño de elite
    elite_size = int(population_size * elite_percentage)
    if elite_size < 1:
        elite_size = 1
    
    # PASO 1: Inicializar población aleatoria
    population = []
    for i in range(population_size):
        individual = []
        for j in range(n):
            individual.append(random.randint(0, 1))
        population.append(individual)
    
    # Variables para tracking
    best_fitness_history = []
    avg_fitness_history = []
    convergence_generation = None
    best_individual = None
    best_fitness = 0
    
    # PASO 2-6: Iteración evolutiva
    for generation in range(max_generations):
        # Evaluar fitness de toda la población
        fitness_scores = []
        for individual in population:
            fitness_scores.append(onemax(individual))
        
        # Encontrar mejor individuo de esta generación
        current_best_fitness = fitness_scores[0]
        current_best_idx = 0
        for i in range(1, len(fitness_scores)):
            if fitness_scores[i] > current_best_fitness:
                current_best_fitness = fitness_scores[i]
                current_best_idx = i
        
        # Actualizar mejor global
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = []
            for gene in population[current_best_idx]:
                best_individual.append(gene)
        
        # Registrar estadísticas
        best_fitness_history.append(best_fitness)
        
        sum_fitness = 0
        for f in fitness_scores:
            sum_fitness += f
        avg_fitness = sum_fitness / len(fitness_scores)
        avg_fitness_history.append(avg_fitness)
        
        # Verificar convergencia (alcanzó el óptimo)
        if best_fitness == n and convergence_generation is None:
            convergence_generation = generation + 1
        
        # Si alcanzamos el óptimo, podemos continuar o terminar
        # (el documento pide 100 generaciones, así que continuamos)
        
        # SELECCIÓN: ordenar población por fitness (de mayor a menor)
        # Crear lista de tuplas (individuo, fitness)
        population_with_fitness = []
        for i in range(len(population)):
            population_with_fitness.append([population[i], fitness_scores[i]])
        
        # Ordenar por fitness descendente
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Seleccionar elite
        elite = []
        for i in range(elite_size):
            elite.append(population_with_fitness[i][0])
        
        # ESTIMACIÓN DE PROBABILIDADES MARGINALES
        # p_i = probabilidad de que x_i = 1 en la elite
        marginal_probabilities = []
        for i in range(n):
            count_ones = 0
            for individual in elite:
                count_ones += individual[i]
            p_i = count_ones / elite_size
            marginal_probabilities.append(p_i)
        
        # GENERACIÓN DE NUEVA POBLACIÓN
        # Muestrear cada variable según su probabilidad marginal
        new_population = []
        for i in range(population_size):
            new_individual = []
            for j in range(n):
                # Muestrear de distribución Bernoulli(p_j)
                sample = bernoulli.rvs(marginal_probabilities[j])
                new_individual.append(sample)
            new_population.append(new_individual)
        
        population = new_population
    
    # Resultado final
    result = {
        'best_individual': best_individual,
        'best_fitness': best_fitness,
        'optimal_value': n,
        'convergence_generation': convergence_generation,
        'best_fitness_history': best_fitness_history,
        'avg_fitness_history': avg_fitness_history,
        'total_generations': max_generations
    }
    
    return result


# ============================================================================
# FUNCIÓN PARA EJECUTAR MÚLTIPLES EXPERIMENTOS
# ============================================================================
def run_experiments(n, population_size, elite_percentage, max_generations, num_runs):
    """
    Ejecuta múltiples ejecuciones independientes del UMDA.
    
    Args:
        n: dimensión del problema
        population_size: tamaño de población
        elite_percentage: porcentaje de elite
        max_generations: generaciones máximas
        num_runs: número de ejecuciones independientes
        
    Returns:
        diccionario con estadísticas agregadas
    """
    print(f"\n{'='*70}")
    print(f"Configuración: n={n}, N={population_size}, Elite={int(elite_percentage*100)}%")
    print(f"{'='*70}")
    
    best_fitness_list = []
    convergence_gen_list = []
    
    for run in range(num_runs):
        result = umda_discreto(n, population_size, elite_percentage, max_generations, seed=run)
        best_fitness_list.append(result['best_fitness'])
        
        if result['convergence_generation'] is not None:
            convergence_gen_list.append(result['convergence_generation'])
        
        # Mostrar progreso cada 5 ejecuciones
        if (run + 1) % 5 == 0:
            print(f"  Ejecución {run + 1}/{num_runs} completada")
    
    # Calcular estadísticas
    # Mejor fitness promedio
    sum_best = 0
    for bf in best_fitness_list:
        sum_best += bf
    mean_best_fitness = sum_best / len(best_fitness_list)
    
    # Desviación estándar del fitness
    sum_squared_diff = 0
    for bf in best_fitness_list:
        sum_squared_diff += (bf - mean_best_fitness) ** 2
    std_best_fitness = math.sqrt(sum_squared_diff / len(best_fitness_list))
    
    # Tasa de éxito (alcanzó el óptimo)
    success_count = 0
    for bf in best_fitness_list:
        if bf == n:
            success_count += 1
    success_rate = (success_count / num_runs) * 100
    
    # Promedio de generaciones hasta convergencia (solo de los exitosos)
    mean_convergence_gen = None
    if len(convergence_gen_list) > 0:
        sum_conv = 0
        for cg in convergence_gen_list:
            sum_conv += cg
        mean_convergence_gen = sum_conv / len(convergence_gen_list)
    
    # Resultados
    results = {
        'n': n,
        'population_size': population_size,
        'num_runs': num_runs,
        'mean_best_fitness': mean_best_fitness,
        'std_best_fitness': std_best_fitness,
        'success_rate': success_rate,
        'mean_convergence_generation': mean_convergence_gen,
        'best_fitness_list': best_fitness_list
    }
    
    # Imprimir resultados
    print(f"\n--- RESULTADOS ---")
    print(f"Mejor valor alcanzado (promedio): {mean_best_fitness:.2f} / {n}")
    print(f"Desviación estándar: {std_best_fitness:.4f}")
    print(f"Tasa de éxito (óptimo alcanzado): {success_rate:.1f}%")
    if mean_convergence_gen is not None:
        print(f"Generaciones promedio hasta convergencia: {mean_convergence_gen:.2f}")
    else:
        print(f"Generaciones promedio hasta convergencia: N/A (no convergió)")
    
    return results


# ============================================================================
# DISEÑO EXPERIMENTAL COMPLETO
# ============================================================================
def run_full_experimental_design():
    """
    Ejecuta el diseño experimental completo según el documento:
    - Tamaños de problema: n = 20, 50, 100
    - Tamaños de población: N = 30, 50, 100
    - Elite: 40%
    - Generaciones: 100
    - Ejecuciones por configuración: 30
    """
    problem_sizes = [20, 50, 100]
    population_sizes = [30, 50, 100]
    elite_percentage = 0.4
    max_generations = 100
    num_runs = 30
    
    print("\n" + "="*70)
    print("DISEÑO EXPERIMENTAL COMPLETO - UMDA DISCRETO")
    print("Función objetivo: OneMax")
    print("="*70)
    
    all_results = []
    
    for n in problem_sizes:
        for pop_size in population_sizes:
            result = run_experiments(n, pop_size, elite_percentage, max_generations, num_runs)
            all_results.append(result)
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN GENERAL DE TODOS LOS EXPERIMENTOS")
    print("="*70)
    print(f"{'n':>5} {'N':>5} {'Mejor (μ)':>12} {'Std':>10} {'Éxito %':>10} {'Gen Conv':>10}")
    print("-"*70)
    
    for res in all_results:
        conv_str = f"{res['mean_convergence_generation']:.1f}" if res['mean_convergence_generation'] else "N/A"
        print(f"{res['n']:>5} {res['population_size']:>5} {res['mean_best_fitness']:>12.2f} "
              f"{res['std_best_fitness']:>10.4f} {res['success_rate']:>9.1f}% {conv_str:>10}")
    
    return all_results


# ============================================================================
# EJECUCIÓN DEL PROGRAMA
# ============================================================================
if __name__ == "__main__":
    # Ejecutar el diseño experimental completo
    results = run_full_experimental_design()
    
    print("\n" + "="*70)
    print("EXPERIMENTOS COMPLETADOS")
    print("="*70)
    
    # Opcional: ejecutar un ejemplo individual para ver la evolución
    print("\n\nEjemplo de ejecución individual (n=20, N=50):")
    print("-"*70)
    example_result = umda_discreto(n=20, population_size=50, elite_percentage=0.4, 
                                   max_generations=100, seed=42)
    
    print(f"Mejor individuo encontrado: {example_result['best_individual']}")
    print(f"Fitness final: {example_result['best_fitness']}/{example_result['optimal_value']}")
    print(f"Convergencia en generación: {example_result['convergence_generation']}")
    
    print("\nEvolución del mejor fitness por generación (primeras 20):")
    for i in range(min(20, len(example_result['best_fitness_history']))):
        gen = i + 1
        fitness = example_result['best_fitness_history'][i]
        bar = '█' * int(fitness)
        print(f"Gen {gen:3d}: {fitness:3d} {bar}")