import os
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

def get_random_function(depth):
    if depth == 0:
        return np.random.choice(variables)
    else:
        op = np.random.choice(operators)
        if op in unary_operators:
            sub = get_random_function(depth-1)
            return f"{op}({sub})"
        else:
            left = get_random_function(depth-1)
            right = get_random_function(depth-1)
            return f"({left} {op} {right})"

def fill_layer(f):
    layer = []
    for l in range(1, DEV_IMAGE_LENGTH + 1):
        for w in range(1, DEV_IMAGE_WIDTH + 1):
            layer.append(eval(f.replace('z', str(w / DEV_IMAGE_WIDTH)).replace('v', str(l / DEV_IMAGE_LENGTH))))
    layer = [round(255 * (i - min(layer)) / (max(layer) - min(layer))) for i in layer]
    return layer

def get_layer():
    try:
        f = get_random_function(np.random.randint(3,7))
        layer = fill_layer(f)
        print(f)
        return f, layer
    except Exception:
        return get_layer()

def generate_image():
    fr, r = get_layer()
    fg, g = get_layer()
    fb, b = get_layer()
    image = []
    for column in range(DEV_IMAGE_LENGTH):
        row_values = []
        for row in range(DEV_IMAGE_WIDTH):
            row_values.append(
                np.array([r[DEV_IMAGE_WIDTH * column + row],
                          g[DEV_IMAGE_WIDTH * column + row],
                          b[DEV_IMAGE_WIDTH * column + row]]))
        image.append(np.array(row_values))
    return [fr, fg, fb], image

def fitness(index_generation):
    scores = list(map(int, input(f'Введите оценки изображений {index_generation} поколения:\n').split()))
    if len(scores) != POPULATION_SIZE:
        print(f'Введено не {POPULATION_SIZE} оценок')
        return fitness(index_generation)
    norm_scores = [(i/sum(scores)) for i in scores]
    return norm_scores

def reproduction(vector, main_population, f):
    cum_weights = np.cumsum(vector)
    spin = np.random.random(size=POPULATION_SIZE) * cum_weights[-1]
    indices = np.searchsorted(cum_weights, spin)
    return [f[i] for i in indices], [main_population[i] for i in indices]

def crossing_over(main_population, f):
    indexes_population_and_f = [i for i in range(len(f))]
    np.random.shuffle(indexes_population_and_f)
    for pair in range(0, len(indexes_population_and_f), 2):
        if np.random.random() < CROSSING_OVER_PROBABILITY:
            gens_for_swap = np.random.randint(0, 3, 2)
            for l in range(DEV_IMAGE_LENGTH):
                for w in range(DEV_IMAGE_WIDTH):
                    main_population[indexes_population_and_f[pair]][l][w][gens_for_swap[0]], \
                        main_population[indexes_population_and_f[pair+1]][l][w][gens_for_swap[1]] = \
                        main_population[indexes_population_and_f[pair+1]][l][w][gens_for_swap[1]], \
                            main_population[indexes_population_and_f[pair]][l][w][gens_for_swap[0]]
            f[indexes_population_and_f[pair]][gens_for_swap[0]], \
                f[indexes_population_and_f[pair+1]][gens_for_swap[1]] = \
                f[indexes_population_and_f[pair+1]][gens_for_swap[1]], \
                    f[indexes_population_and_f[pair]][gens_for_swap[0]]
    return f, main_population

def mutate(main_population, f):
    for index, individual in enumerate(main_population):
        if np.random.random() < MUTATION_PROBABILITY:
            gens_for_mutate = np.random.randint(0, 3)
            while True:
                operators_pull = [unary_operators, binary_operators[:-2]]
                current_operator_kind = operators_pull[np.random.randint(0, 2)]
                first_operator = np.random.choice(current_operator_kind)
                second_operator = np.random.choice(current_operator_kind)
                if first_operator != second_operator and first_operator in f[index][gens_for_mutate]:
                    new_formula = f[index][gens_for_mutate].replace(first_operator, second_operator)
                    try:
                        new_layer = fill_layer(new_formula)
                    except Exception:
                        continue
                    counter_layer_value = 0
                    for l in range(DEV_IMAGE_LENGTH):
                        for w in range(DEV_IMAGE_WIDTH):
                            main_population[index][l][w][gens_for_mutate] =  new_layer[counter_layer_value]
                            counter_layer_value += 1
                    f[index][gens_for_mutate] = new_formula
                    break
    return f, main_population

def enlarge_and_save_population(main_population, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for index, individual in enumerate(main_population, start=1):
        image = Image.fromarray(np.array(individual).astype(np.uint8))
        resized_image = image.resize((FINAL_IMAGE_WIDTH, FINAL_IMAGE_LENGTH), Image.BILINEAR)
        resized_image.save(f"{path}/image_{index}.png")

if __name__ ==  '__main__':
    unary_operators = ["np.sin", "np.cos", "np.exp", "np.tan", "np.log", "np.sinh", "np.cosh", "np.tanh", "abs"]
    binary_operators = ["+", "-", "**", "//", "%", "*", "/"]
    operators = unary_operators + binary_operators
    variables = ["z", "v"]

    DEV_IMAGE_WIDTH = 50
    DEV_IMAGE_LENGTH = 100

    FINAL_IMAGE_WIDTH = DEV_IMAGE_WIDTH * 15
    FINAL_IMAGE_LENGTH = DEV_IMAGE_LENGTH * 15

    POPULATION_SIZE = 10
    NUMBER_OF_GENERATIONS = 10
    MUTATION_PROBABILITY = 1
    CROSSING_OVER_PROBABILITY = 0.8

    population_and_f = [generate_image() for i in range(POPULATION_SIZE)]
    formulas = [i[0] for i in population_and_f]
    population = [i[1] for i in population_and_f]
    enlarge_and_save_population(population, 'images/initial')
    for i in range(NUMBER_OF_GENERATIONS):
        norm_vector = fitness(i)
        formulas, population = reproduction(norm_vector, population, formulas)
        formulas, population = crossing_over(population, formulas)
        formulas, population = mutate(population, formulas)
        enlarge_and_save_population(population, f'images/generation_{i+1}')
