using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ML.NeuralMethods.Model;
using ML.Contracts;
using ML.Core;

namespace ML.Utils
{
  public static class NetworkFactory
  {
    public static NeuralNetwork CreateFullyConnectedNetwork(int[] topology,
                                                            IFunction activationFunction,
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
  }
}
