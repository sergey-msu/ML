using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ML.NeuralMethods.Models;
using ML.DeepMethods.Models;
using ML.Contracts;
using ML.Core;
using ML.Core.Registry;

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

    /// <summary>
    /// Creates CNN with original LeNet-1 architecture (see http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf)
    /// </summary>
    public static ConvolutionalNetwork CreateLeNet1(IActivationFunction activation = null,
                                                    bool randomizeInitialWeights = true,
                                                    int randomSeed = 0)
    {
      activation = activation ?? Activation.ReLU;
      var net = new ConvolutionalNetwork(1, 28) { IsTraining=true };

      net.AddLayer(new ConvolutionalLayer(outputDepth: 8, windowSize: 5, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new ConvolutionalLayer(outputDepth: 18, windowSize: 5, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new FlattenLayer(outputDim: 10, activation: activation));

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
      activation = activation ?? Activation.ReLU;
      var net = new ConvolutionalNetwork(3, 32) { IsTraining=true };

      net.AddLayer(new ConvolutionalLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new ConvolutionalLayer(outputDepth: 32, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new ConvolutionalLayer(outputDepth: 64, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new ConvolutionalLayer(outputDepth: 64, windowSize: 3, padding: 1, activation: activation));
      net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2));
      net.AddLayer(new DropoutLayer(0.25));

      net.AddLayer(new FlattenLayer(outputDim: 512, activation: Activation.ReLU));
      net.AddLayer(new DropoutLayer(0.5));
      net.AddLayer(new DenseLayer(outputDim: 10, activation: Activation.Logistic(1)));

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
