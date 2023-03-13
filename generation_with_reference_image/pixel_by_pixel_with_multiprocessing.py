import numpy as np
import multiprocessing
from PIL import Image
import time


def get_initial_generation(population_size, chromosomes_number):
    return [''.join(list(map(lambda x: str(x),
                             list(np.random.randint(2, size=chromosomes_number)))))
            for _ in range(population_size)]

def fitness(bin_population, target_value):
    epsilon = 1e-10
    dec_population_value = list(map(lambda x: int(x, 2), bin_population))
    deviation_vector = [abs(target_value-i) for i in dec_population_value]
    pre_norm_vector = [1 / (d+epsilon) for d in deviation_vector]
    norm_values = [i/sum(pre_norm_vector) for i in pre_norm_vector]
    return norm_values

def reproduction(normalized_vector, bin_population, population_size):
    cum_weights = np.cumsum(normalized_vector)
    spin = np.random.random(size=population_size) * cum_weights[-1]
    indices = np.searchsorted(cum_weights, spin)
    return [bin_population[i] for i in indices]

def crossing_over(bin_population, crossing_over_probability):
    np.random.shuffle(bin_population)
    for i in range(0, len(bin_population), 2):
        if np.random.random() < crossing_over_probability:
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

def mutation(bin_population, mutation_probability):
    for i in range(len(bin_population)):
        if np.random.random() < mutation_probability:
            number_of_gen = np.random.randint(len(bin_population[i]))
            bin_population[i] = bin_population[i][:number_of_gen] + \
                                str((int(bin_population[i][number_of_gen]) + 1) % 2) + \
                                bin_population[i][number_of_gen + 1:]
    return bin_population

def evolve_image(target_row):
    POPULATION_SIZE = 18
    NUMBER_OF_GENERATIONS = 40

    MUTATION_PROBABILITY = 0.05
    CROSSING_OVER_PROBABILITY = 0.5
    CHROMOSOMES_NUMBER = 8
    row_result = []
    for target_pixel in target_row:
        result_pixel = []
        for channel in target_pixel:
            population = get_initial_generation(POPULATION_SIZE, CHROMOSOMES_NUMBER)
            for generation in range(NUMBER_OF_GENERATIONS):
                norm_vector = fitness(population, channel)
                population = reproduction(norm_vector, population, POPULATION_SIZE)
                population = crossing_over(population, CROSSING_OVER_PROBABILITY)
                population = mutation(population, MUTATION_PROBABILITY)
            norm_vector = fitness(population, channel)
            result_pixel.append(population[norm_vector.index(max(norm_vector))])
        result_pixel = np.array(list(map(lambda x: int(x, 2), result_pixel)))
        row_result.append(np.array(result_pixel))
    print('DONE')
    return row_result


if __name__ == '__main__':
    start_time = time.time()
    TARGET_IMAGE = 'images/target.png'
    target_image = Image.open(TARGET_IMAGE).convert('RGBA')
    target_array = np.array(target_image)

    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
        result_image = pool.map(evolve_image, target_array)

    Image.fromarray(np.array(result_image).astype(np.uint8)).save(f"images/best_image.png")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Программа выполнялась", elapsed_time, "секунд.")
