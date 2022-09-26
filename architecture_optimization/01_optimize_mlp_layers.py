from deap import base, creator, tools
import random
import numpy as np
import mlp_layers_test
import elitism


# boundaries for layer size parameters:
# [hidden_layer_1_size, [hidden_layer_2_size, [hidden_layer_3_size, [hidden_layer_4_size]
# for example the first hidden layer is given the range [5, 15]
BOUNDS_LOW = [5, -5, -10, -20]
BOUNDS_HIGH = [15, 10, 10, 10]

NUM_OF_PARAMS = len(BOUNDS_HIGH)

# Genetic algorithm constants:
POPULATION_SIZE = 20
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5  # probability for mutating an individual
MAX_GENERATIONS = 10
HALL_OF_FAME_SIZE = 3
CROWDING_FACTOR = 10.0  # crowding factor for crossover and mutation

# set the random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the classifier accuracy test class
test = mlp_layers_test.MlpLayersTest(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, maximizing fitness strategy
# our goal is to maximize the accuracy of the classifier.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# define the layer_size_attributes individually:
# since the solution is represented by a list of float values, each of a different
# range, we use the following loop to iterate over all pairs of lower-bound,
# upper bound values, and for each range, we create a separate toolbox operator
# ("layer_size_attribute") that will be used to generate random float values in
# the appropriate range later:
for i in range(NUM_OF_PARAMS):
    # "layer_size_attribute_0", "layer_size_attribute_1", ...
    toolbox.register(
        "layer_size_attribute_" + str(i), random.uniform, BOUNDS_LOW[i], BOUNDS_HIGH[i]
    )


# create a tuple containing a layer_size_attribute generator for each hidden layer:
layer_size_attributes = ()
for i in range(NUM_OF_PARAMS):
    layer_size_attributes = layer_size_attributes + (
        toolbox.__getattribute__("layer_size_attribute_" + str(i)),
    )

# create the individual operator to fill up an Individual instance:
# we use layer_size_attributes tuple in conjuction with DEAP's built-in initCycle
# operator to create a new individualCreator operator that fills up an individual
# instance with a combination of randomly generated hidden layer size values
toolbox.register(
    "individualCreator", tools.initCycle, creator.Individual, layer_size_attributes, n=1
)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# fitness calculation
def classification_accuracy(individual):
    return test.get_accuracy(individual)


toolbox.register("evaluate", classification_accuracy)

# genetic operators:mutFlipBit

toolbox.register("select", tools.selTournament, tournsize=2)

toolbox.register(
    "mate",
    tools.cxSimulatedBinaryBounded,
    low=BOUNDS_LOW,
    up=BOUNDS_HIGH,
    eta=CROWDING_FACTOR,
)

toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=BOUNDS_LOW,
    up=BOUNDS_HIGH,
    eta=CROWDING_FACTOR,
    indpb=1.0 / NUM_OF_PARAMS,
)

# Genetic Algorithm flow
def main():
    # create initial population (generation 0)
    # print("Todo cool hasta aquí")
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    print("Aquí todo va bien")

    # define the hall-of-fame object
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorith flow with hof feature added
    population, logbook = elitism.ea_simple_with_elitism(
        population,
        toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        ngen=MAX_GENERATIONS,
        stats=stats,
        hall_of_fame=hof,
        verbose=True,
    )

    # print the solution found
    print(
        "- Best solution is: ",
        test.formatParams(hof.items[0]),
        ", accuracy = ",
        hof.items[0].fitness.values[0],
    )


if __name__ == "__main__":
    main()
