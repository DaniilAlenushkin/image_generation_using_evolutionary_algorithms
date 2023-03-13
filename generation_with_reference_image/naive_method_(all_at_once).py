from multiprocessing import Pool
import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

# Задание параметров эволюционного алгоритма
TARGET_IMAGE = 'images/target.png'
POPULATION_SIZE = 1000
GENERATIONS = 500000
MUTATION_PROBABILITY = 0.05
CROSSING_OVER_PROBABILITY = 0.5

# Загрузка эталонного изображения и преобразование в массив numpy
target_image = Image.open(TARGET_IMAGE).convert('RGBA')
target_array = np.array(target_image)

# Создание случайного изображения
def create_random_image():
    width, height, channels = target_array.shape
    return np.random.rand(height, width, channels) * 255

# Вычисление "фитнеса" изображения, т.е. степени его близости к эталонному
def fitness(image):
    mse = mean_squared_error(target_array, image)
    ssim = structural_similarity(target_array, image, multichannel=True, win_size=3, data_range=255)
    psnr = peak_signal_noise_ratio(target_array, image)
    return mse + (1 - ssim) + (1 / psnr)


# Мутация изображения
def mutate(image):
    if np.random.random() < MUTATION_PROBABILITY:
        height, width, channels = image.shape
        num_pixels_to_mutate = np.random.randint(0, (int(width*height)/10))
        pixels_to_mutate = np.random.choice(width*height, num_pixels_to_mutate, replace=False)
        y, x = np.unravel_index(pixels_to_mutate, (height, width))
        image[y, x, :] = np.random.rand(num_pixels_to_mutate, 4) * 255
    return image

def crossover(image1, image2):
    if np.random.random() < CROSSING_OVER_PROBABILITY:
        height, width, channels = image1.shape
        y = np.random.randint(0, height - 1)
        x = np.random.randint(0, width - 1)
        if np.random.random() < 0.5:
            child1 = np.concatenate((image1[:y, :, :], image2[y:, :, :]), axis=0)
            child2 = np.concatenate((image2[:y, :, :], image1[y:, :, :]), axis=0)
        else:
            child1 = np.concatenate((image1[:, :x, :], image2[:, x:, :]), axis=1)
            child2 = np.concatenate((image2[:, :x, :], image1[:, x:, :]), axis=1)
        image1, image2 = child1, child2
    return image1, image2


def reproduction(vector, population):
    cum_weights = np.cumsum(vector)
    spin = np.random.random(size=POPULATION_SIZE) * cum_weights[-1]
    indices = np.searchsorted(cum_weights, spin)
    return [population[i] for i in indices]

# Выполнение эволюции популяции в нескольких процессах
def evolve_population():
    population = [create_random_image() for _ in range(POPULATION_SIZE)]
    pool = Pool(processes=None)
    for i in range(GENERATIONS):
        # Вычисляем значения фитнес-функции
        fitness_pop = pool.map(fitness, population)
        # Нормируем эти значения
        fitness_pop_prenorm = [1 - (i / sum(fitness_pop)) for i in fitness_pop]
        fitness_pop_norm = [i / sum(fitness_pop_prenorm) for i in fitness_pop_prenorm]
        # Репродукция
        population = reproduction(fitness_pop_norm, population)
        # Кроссинг-овер
        np.random.shuffle(population)
        new_population = []
        for pair in range(0, len(population), 2):
            child1, child2 = crossover(population[pair], population[pair + 1])
            new_population.extend([child1, child2])
        population = pool.map(mutate, new_population[:])
        best_image = min(population, key=fitness)
        print(f"Generation {i + 1}: Best fitness = {fitness(best_image)}")
        if (i+1)%1000 == 0:
            Image.fromarray(best_image.astype(np.uint8)).save(f"images/best_image_{i+1}_{fitness(best_image)}.png")

# Основной код
if __name__ == '__main__':
    evolve_population()
