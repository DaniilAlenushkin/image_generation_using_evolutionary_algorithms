import numpy as np
from PIL import Image
import time

def get_initial_generation():
    return [''.join(list(map(lambda x: str(x),
                             list(np.random.randint(2, size=CHROMOSOMES_NUMBER)))))
            for _ in range(POPULATION_SIZE)]

def fitness(bin_population, target_value):
    epsilon = 1e-10
    dec_population_value = list(map(lambda x: int(x, 2), bin_population))
    deviation_vector = [abs(target_value-i) for i in dec_population_value]
    pre_norm_vector = [1 / (d+epsilon) for d in deviation_vector]
    norm_values = [i/sum(pre_norm_vector) for i in pre_norm_vector]
    return norm_values

def reproduction(normalized_vector, bin_population):
    cum_weights = np.cumsum(normalized_vector)
    spin = np.random.random(size=POPULATION_SIZE) * cum_weights[-1]
    indices = np.searchsorted(cum_weights, spin)
    return [bin_population[i] for i in indices]

def crossing_over(bin_population):
    np.random.shuffle(bin_population)
    for i in range(0, len(bin_population), 2):
        if np.random.random() < CROSSING_OVER_PROBABILITY:
            number_of_genes = np.random.randint(len(bin_population[0]))
            direction = np.random.randint(2)
            if direction:  # faces
                face_1 = bin_population[i][:number_of_genes]
                face_2 = bin_population[i + 1][:number_of_genes]
                bin_population[i] = face_2 + bin_population[i][number_of_genes:]
                bin_population[i + 1] = face_1 + bin_population[i + 1][number_of_genes:]
            else:  # back
                back_1 = bin_population[i][len(bin_population[i]) - number_of_genes:]
                back_2 = bin_population[i + 1][len(bin_population[i]) - number_of_genes:]
                bin_population[i] = bin_population[i][:len(bin_population[i]) - number_of_genes] + back_2
                bin_population[i + 1] = bin_population[i + 1][:len(bin_population[i]) - number_of_genes] + back_1
    return bin_population

def mutation(bin_population):
    for i in range(len(bin_population)):
        if np.random.random() < MUTATION_PROBABILITY:
            number_of_gen = np.random.randint(len(bin_population[i]))
            bin_population[i] = bin_population[i][:number_of_gen] + \
                                str((int(bin_population[i][number_of_gen]) + 1) % 2) + \
                                bin_population[i][number_of_gen + 1:]
    return bin_population

if __name__ == '__main__':
    start_time = time.time()
    TARGET_IMAGE = 'images/target.png'
    POPULATION_SIZE = 100
    NUMBER_OF_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.05
    CROSSING_OVER_PROBABILITY = 0.5
    GENS_NUMBER = 4
    CHROMOSOMES_NUMBER = 8

    target_image = Image.open(TARGET_IMAGE).convert('RGBA')
    target_array = np.array(target_image)

    height, width, channels = target_array.shape
    result_image = []
    # сначала по строкам
    for h in range(1, height+1):
        row_result = []
        # Потом для каждого пикселя в строке
        for w in range(1, width+1):
            result_pixel = []
            # Для каждой компоненты
            for c in range(channels):
                target = target_array[h-1:h, w-1:w,:][0][0][c]
                population = get_initial_generation()
                for generation in range(NUMBER_OF_GENERATIONS):
                    norm_vector = fitness(population, target)
                    population = reproduction(norm_vector, population)
                    population = crossing_over(population)
                    population = mutation(population)
                norm_vector = fitness(population, target)
                result_pixel.append(population[norm_vector.index(max(norm_vector))])
            result_pixel = np.array(list(map(lambda x: int(x, 2), result_pixel)))
            print(h, w, 'DONE')
            row_result.append(np.array(result_pixel))
        result_image.append(np.array(row_result))
    Image.fromarray(np.array(result_image).astype(np.uint8)).save(f"images/best_image.png")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Программа выполнялась", elapsed_time, "секунд.")


