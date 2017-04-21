using System;
using ML.NeuralMethods.Models;
using ML.DeepMethods.Models;
using ML.Contracts;
using ML.Core;

namespace ML.Utils
{
  public static class NetworkFactory
  {
    /// <summary>
    /// Creates fully connected NN
    /// </summary>
    /// <param name="topology">Network topology from input to output layer
    /// (i.e. [2,10,3] means NN with 2D input, 1 hidden layer with 10 neurons and 3D output)</param>
    public static NeuralNetwork CreateFullyConnectedNetwork(int[] topology,
                                                            IActivationFunction activation = null,
                                                            bool randomizeInitialWeights = true,
                                                            int randomSeed = 0)
    {
      if (topology==null || topology.Length<2)
        throw new MLException("Network topology must have at least input and output dimensions");

      var net = new NeuralNetwork(topology[0], activation);

      var lcount = topology.Length-1;
      for (int i=1; i<=lcount; i++)
      {
        var neuronCount = topology[i];
        var layer = new NeuralLayer(neuronCount);
        net.AddLayer(layer);
      }

      net.Build();

      if (randomizeInitialWeights)
        net.RandomizeParameters(randomSeed);

      return net;
    }

    // create VGG16 CNN (see https://arxiv.org/pdf/1409.1556.pdf - D column in Table 1)

    // create AlexNet

    // create U-net

    // create general N-M-K ConvNet :  INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
  }
}
