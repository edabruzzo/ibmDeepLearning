import numpy as np # import Numpy library to generate

class Teste1(object):


    def testing_simple_neural_net(self):

        weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
        biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases
        print(weights)
        print(biases)

        x_1 = 0.5 # input 1
        x_2 = 0.85 # input 2
        print('x1 is {} and x2 is {}'.format(x_1, x_2))

        z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
        print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))

        z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
        print('The weighted sum of the inputs at the second node in the hidden layer is {}'
               .format(np.around(z_12, decimals=4)))

        a_11 = 1.0 / (1.0 + np.exp(-z_11))
        print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))

        a_12 = 1.0 / (1.0 + np.exp(-z_12))
        print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))

        z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
        print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))

        a_2 = 1.0 / (1.0 + np.exp(-z_2))
        print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))



    def configure_network_many_hidden_layers(self, num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):


        num_nodes_previous = num_inputs  # number of nodes in the previous layer


        network = {}  # initialize network as an empty dictionary

        # loop through each layer and randomly initialize the weights and biases associated with each node
        # notice how we are adding 1 to the number of hidden layers in order to include the output layer
        for layer in range(num_hidden_layers + 1):

            # determine name of layer
            if layer == num_hidden_layers:
                layer_name = 'output'
                num_nodes = num_nodes_output
            else:
                layer_name = 'layer_{}'.format(layer + 1)
                num_nodes = num_nodes_hidden[layer]

            # initialize weights and biases associated with each node in the current layer
            network[layer_name] = {}
            for node in range(num_nodes):
                node_name = 'node_{}'.format(node + 1)
                network[layer_name][node_name] = {
                    'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                    'bias': np.around(np.random.uniform(size=1), decimals=2),
                }

            num_nodes_previous = num_nodes

        return network  # print network


    def compute_weighted_sum(self, inputs, weights, bias):
        return np.sum(inputs * weights) + bias

    def node_activation(self, weighted_sum):
        return 1.0 / (1.0 + np.exp(-1 * weighted_sum))


from random import seed


if __name__ == '__main__':

     #Teste1().testing_simple_neural_net()

     network = Teste1().configure_network_many_hidden_layers(5, 3, [3, 2, 3], 1)

     np.random.seed(12)
     inputs = np.around(np.random.uniform(size=5), decimals=2)
     print('The inputs to the network are {}'.format(inputs))

     weights = network['layer_1']['node_1']['weights']
     bias = network['layer_1']['node_1']['bias']
     print(weights)
     print(bias)

     '''
     output = Teste1().compute_weighted_sum(inputs,
                                   network['layer_1']['node_1']['weights'], # weights
                                   network['layer_1']['node_1']['bias']) # bias

     print(output)

     Teste1().node_activation(output)
     
     '''

     activation_sigmoid = Teste1().node_activation(compute_weighted_sum(inputs,
                                   weight = network['layer_1']['node_1']['weights'], # weights
                                   bias = network['layer_1']['node_1']['bias']))


     print(activation_sigmoid)


