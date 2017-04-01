using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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
    /// Creates CNN with original LeNet-1 architecture (see http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf)
    /// </summary>
    public static ConvolutionalNetwork CreateLeNet1(IActivationFunction activation = null,
                                                    bool randomizeInitialWeights = true,
                                                    int randomSeed = 0)
    {
      activation = activation ?? Registry.ActivationFunctions.ReLU;

      var net = new ConvolutionalNetwork(1, 28)
      {
        ActivationFunction = activation,
        IsTraining = true
      };

      net.AddLayer(new ConvolutionalLayer(outputDepth: 4, windowSize: 5));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new ConvolutionalLayer(outputDepth: 12, windowSize: 5));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new FlattenLayer(outputDim: 10));

      net.Build();

      if (randomizeInitialWeights)
        net.RandomizeParameters(randomSeed);

      return net;
    }

    /// <summary>
    /// Creates CNN for CIFAR-10 training
    /// </summary>
    public static ConvolutionalNetwork CreateCIFAR10Net(IActivationFunction activation = null,
                                                        bool randomizeInitialWeights = true,
                                                        int randomSeed = 0)
    {
      activation = activation ?? Registry.ActivationFunctions.ReLU;

      var net = new ConvolutionalNetwork(3, 32)
      {
        ActivationFunction = activation,
        IsTraining = true
      };

      net.AddLayer(new ConvolutionalLayer(outputDepth: 32, windowSize: 3, padding: 1));
      net.AddLayer(new ConvolutionalLayer(outputDepth: 32, windowSize: 3, padding: 1));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new ConvolutionalLayer(outputDepth: 64, windowSize: 3, padding: 1));
      net.AddLayer(new ConvolutionalLayer(outputDepth: 64, windowSize: 3, padding: 1));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new FlattenLayer(outputDim: 512));
      net.AddLayer(new DenseLayer(outputDim: 10) { ActivationFunction=Registry.ActivationFunctions.Logistic(1) });

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
