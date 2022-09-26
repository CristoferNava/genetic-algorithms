"""
To be able to use our standard genetic operators, we will use a fixed-length 
representaion of the number of layers. When using this approach, the maximum 
number of layers is decided in advance, and all the layers are always represented,
but not necessarily expressed in the solution. For example, if we decide to limit
the network to four hidden layers, the chromosome will look as follows:
[n1, n2, n3, n4]
here n_i denotes the number of nodes in the layer i.

To control the actual number of hidden layers in the network, some of these values
may be zero, or negative. Such a value means that no more layers will be added to
the network. For example:
[10, 20, -5, 15] 
is translated into the tuple (10, 20) since the -5 terminates the layer count.
"""
from sklearn import model_selection, datasets
from sklearn.neural_network import MLPClassifier

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings


class MlpLayersTest:
    """Class used by the genetic algorithm-based optimizer."""

    NUM_FOLDS = 5

    def __init__(self, random_seed):
        self.random_seed = random_seed
        self.init_dataset()
        self.kfold = model_selection.KFold(
            n_splits=self.NUM_FOLDS, random_state=self.random_seed, shuffle=True
        )

    def init_dataset(self):
        self.data = datasets.load_iris()
        self.X = self.data["data"]
        self.y = self.data["target"]

    def convert_params(self, params):
        """params contains: [layer_1_size, layer_2_size, layer_3_size, layer_4_size]
        this is the chromosome that we described in the previous subsection
        and contains the float values that represent up to four hidden layers.
        The method transforms this list of floats into the hidden_layer_sizes tuple."""
        if round(params[1]) <= 0:
            hidden_layers_sizes = round(params[0])
        elif round(params[2]) <= 0:
            hidden_layers_sizes = (round(params[0]), round(params[1]))
        elif round(params[3]) <= 0:
            hidden_layers_sizes = (
                round(params[0]),
                round(params[1]),
                round(params[2]),
            )
        else:
            hidden_layers_sizes = (
                round(params[0]),
                round(params[1]),
                round(params[2]),
                round(params[3]),
            )

        return hidden_layers_sizes

    @ignore_warnings(category=ConvergenceWarning)
    def get_accuracy(self, params):
        """Takes the params list representing the configuration of the hidden
        layers, uses the convert_param() method to transform it into the
        hidden into the hidden_layers_sizes tuple, and initializes the MLP
        classfier with this tuple. Then it finds the accuracy of the
        classfier using the k-folds cross-validation."""
        hidden_layers_sizes = self.convert_params(params)

        self.classifier = MLPClassifier(
            random_state=self.random_seed, hidden_layer_sizes=hidden_layers_sizes
        )

        cv_results = model_selection.cross_val_score(
            self.classifier, self.X, self.y, cv=self.kfold, scoring="accuracy"
        )
        return cv_results.mean()

    def format_params(self, params):
        return f"hidden_layer_sizes={self.convert_params(params)}"
