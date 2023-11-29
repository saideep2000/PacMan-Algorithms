import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

        w = self.get_weights()
        dot_product = nn.DotProduct(x, w)
        return dot_product

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dot_product = self.run(x)
        floating_point_number = nn.as_scalar(dot_product)

        if floating_point_number >= 0:
            return 1
        else:
            return -1

    def update_for_any_mistakes(self, dataset):
        for x, y in dataset.iterate_once(1):
            multiplier = nn.as_scalar(y)
            if self.get_prediction(x) != multiplier:
                nn.Parameter.update(self.w, x, multiplier)
                return True
        return False

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        mistakes = True
        while mistakes:
            mistakes = self.update_for_any_mistakes(dataset)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size = 90
        self.batch_size = 200
        self.learning_rate = -0.05
        self.loss_recorded = []
        self.total_recorded_instances = 0

        # initializing for 2 layers and 2 biases
        self.w1 = nn.Parameter(1, self.hidden_layer_size)
        self.w2 = nn.Parameter(self.hidden_layer_size, 1)

        self.bias_for_w1 = nn.Parameter(1, self.hidden_layer_size)
        self.bias_for_w2 = nn.Parameter(1, 1)

        self.parameters = [self.w1, self.bias_for_w1, self.w2, self.bias_for_w2]

        # with this got 0.006909 final loss value

        # initializing for 3 layers and 3 biases
        # self.w1 = nn.Parameter(1, self.hidden_layer_size)
        # self.w2 = nn.Parameter(self.hidden_layer_size, 30)
        # self.w3 = nn.Parameter(30, 1)
        #
        # self.bias_for_w1 = nn.Parameter(1, self.hidden_layer_size)
        # self.bias_for_w2 = nn.Parameter(1, 30)
        # self.bias_for_w3 = nn.Parameter(1, 1)
        #
        # self.parameters = [self.w1, self.bias_for_w1, self.w2, self.bias_for_w2, self.w3, self.bias_for_w3]

        # with this got 0.002028 final loss value

        # Though implemented 3 layers, it may over fit sometimes, but we need to strike balance between them.


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        # 1 hidden layer and 1 output layer
        transformation_layer_1 = nn.Linear(x, self.w1)
        bias_to_transformation_layer_1 = nn.AddBias(transformation_layer_1, self.bias_for_w1)
        relu_to_transformation_layer_1 = nn.ReLU(bias_to_transformation_layer_1)

        transformation_layer_output = nn.Linear(relu_to_transformation_layer_1, self.w2)
        bias_to_transformation_layer_output = nn.AddBias(transformation_layer_output, self.bias_for_w2)

        return bias_to_transformation_layer_output

        # 2 hidden layers and 1 output layer
        # hidden layer 1
        # transformation_layer_1 = nn.Linear(x, self.w1)
        # bias_to_transformation_layer_1 = nn.AddBias(transformation_layer_1, self.bias_for_w1)
        # relu_to_transformation_layer_1 = nn.ReLU(bias_to_transformation_layer_1)
        #
        # # hidden layer 2
        # transformation_layer_2 = nn.Linear(relu_to_transformation_layer_1, self.w2)
        # bias_to_transformation_layer_2 = nn.AddBias(transformation_layer_2, self.bias_for_w2)
        # relu_to_transformation_layer_2 = nn.ReLU(bias_to_transformation_layer_2)
        #
        # # Output layer 1
        # transformation_layer_output = nn.Linear(relu_to_transformation_layer_2, self.w3)
        # bias_to_transformation_layer_output = nn.AddBias(transformation_layer_output, self.bias_for_w3)

        # return bias_to_transformation_layer_output



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y_output = self.run(x)
        square_loss_computed = nn.SquareLoss(predicted_y_output, y)
        return square_loss_computed

    def update_weights(self, learning_rate, gradient_values):
        a = 0
        while a < len(gradient_values):
            self.parameters[a].update(gradient_values[a], learning_rate)
            a = a + 1

    def calculate_average(self, loss):
        self.loss_recorded.append(nn.as_scalar(loss))
        average_training_loss = sum(self.loss_recorded) / self.total_recorded_instances
        return average_training_loss
    def update_weights_while_iterating(self, dataset, track_loss):
        square_loss = 0
        for x, y in dataset.iterate_once(self.batch_size):
            square_loss = self.get_loss(x, y)
            loss_value = nn.as_scalar(square_loss)
            if loss_value != track_loss[-1]:
                gradients = nn.gradients(square_loss, self.parameters)
                self.update_weights(self.learning_rate, gradients)
            track_loss.append(loss_value)
        return square_loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        average_training_loss = 1
        square_loss_computed = 0
        track_loss = [0]
        while average_training_loss > 0.02:
            square_loss_computed = square_loss_computed + self.update_weights_while_iterating(dataset, track_loss)
            self.total_recorded_instances = self.total_recorded_instances + 1
            average_training_loss = self.calculate_average(square_loss_computed)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size = 200
        self.batch_size = 100
        self.learning_rate = -0.5
        self.loss_recorded = []
        self.total_recorded_instances = 0

        # initializing for 3 layers and 3 biases
        self.w1 = nn.Parameter(784, self.hidden_layer_size)
        self.w2 = nn.Parameter(self.hidden_layer_size, 100)
        self.w3 = nn.Parameter(100, 10)

        self.bias_for_w1 = nn.Parameter(1, self.hidden_layer_size)
        self.bias_for_w2 = nn.Parameter(1, 100)
        self.bias_for_w3 = nn.Parameter(1, 10)

        self.parameters = [self.w1, self.bias_for_w1, self.w2, self.bias_for_w2, self.w3, self.bias_for_w3]

        # with this got 0.002028 final loss value


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # 2 hidden layers and 1 output layer
        # hidden layer 1
        transformation_layer_1 = nn.Linear(x, self.w1)
        bias_to_transformation_layer_1 = nn.AddBias(transformation_layer_1, self.bias_for_w1)
        relu_to_transformation_layer_1 = nn.ReLU(bias_to_transformation_layer_1)

        # hidden layer 2
        transformation_layer_2 = nn.Linear(relu_to_transformation_layer_1, self.w2)
        bias_to_transformation_layer_2 = nn.AddBias(transformation_layer_2, self.bias_for_w2)
        relu_to_transformation_layer_2 = nn.ReLU(bias_to_transformation_layer_2)

        # Output layer 1
        transformation_layer_output = nn.Linear(relu_to_transformation_layer_2, self.w3)
        bias_to_transformation_layer_output = nn.AddBias(transformation_layer_output, self.bias_for_w3)

        return bias_to_transformation_layer_output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y_output = self.run(x)
        softmax_loss_computed = nn.SoftmaxLoss(predicted_y_output, y)
        return softmax_loss_computed

    def update_weights(self, learning_rate, gradient_values):
        a = 0
        while a < len(gradient_values):
            self.parameters[a].update(gradient_values[a], learning_rate)
            a = a+1
    def update_weights_while_iterating(self, dataset, track_loss):
        for x, y in dataset.iterate_once(self.batch_size):
            loss = self.get_loss(x, y)
            loss_value = nn.as_scalar(loss)
            if loss_value != track_loss[-1]:
                gradients = nn.gradients(loss, self.parameters)
                self.update_weights(self.learning_rate, gradients)
            track_loss.append(loss_value)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        validation_accuracy = 0
        track_loss = [0]
        while validation_accuracy < 0.98:
            self.update_weights_while_iterating(dataset, track_loss)
            validation_accuracy = dataset.get_validation_accuracy()



class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"