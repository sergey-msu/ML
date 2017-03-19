using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ML.NeuralMethods.Model;
using ML.DeepMethods.Model;
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
    /// <returns></returns>
    public static NeuralNetwork CreateFullyConnectedNetwork(int[] topology,
                                                            IActivationFunction activationFunction,
                                                            bool randomizeInitialWeights = true,
                                                            int randomSeed = 0)
    {
      if (topology==null || topology.Length<2)
        throw new MLException("Network topology must have at least input and output dimensions");
      if (activationFunction==null)
        throw new MLException("Activation function is null");

      var net = new NeuralNetwork(topology[0]);
      net.ActivationFunction = activationFunction;

      var lcount = topology.Length-1;
      for (int i=1; i<=lcount; i++)
      {
        var pdim = topology[i-1];
        var dim = topology[i];
        var layer = new NeuralLayer(pdim);
        net.AddLayer(layer);
        for (int j=0; j<dim; j++)
        {
          var neuron = new Neuron(layer.InputDim);
          layer.AddNeuron(neuron);
        }
      }

      net.Build();

      if (randomizeInitialWeights)
        net.RandomizeParameters(randomSeed);

      return net;
    }

    /// <summary>
    /// Sreate CNN with original LeNet-1 architecture (see http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf)
    /// </summary>
    /// <returns></returns>
    public static ConvolutionalNetwork CreateLeNet1Network(IActivationFunction activationFunction = null)
    {
      var net = new ConvolutionalNetwork(1, 28);

      var layer1 = new ConvolutionalLayer(1, 28, 4, 5, 1, isTraining: true);
      net.AddLayer(layer1);

      var layer2 = new MaxPoolingLayer(4, 24, 2, 2, isTraining: true);
      net.AddLayer(layer2);

      var layer3 = new ConvolutionalLayer(4, 12, 12, 5, 1, isTraining: true);
      net.AddLayer(layer3);

      var layer4 = new MaxPoolingLayer(12, 8, 2, 2, isTraining: true);
      net.AddLayer(layer4);

      var layer5 = new ConvolutionalLayer(12, 4, 10, 4, 1, isTraining: true);
      layer1.ActivationFunction = Registry.ActivationFunctions.Rational(2);
      net.AddLayer(layer5);

      net.ActivationFunction = activationFunction ?? Registry.ActivationFunctions.Identity;

      net.Build();

      return net;
    }

    // create AlexNet

    // create N-M-K ConvNet :  INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
  }
}
