using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.NeuralMethods.Model
{
  /// <summary>
  /// Represents feedforward artificial neural network: set of layers with neuron nodes and weighted connections
  /// </summary>
  public class NeuralNetwork : NetworkNode<double, NeuralLayer, Neuron>
  {
    public NeuralNetwork(int inputDim) : base(inputDim)
    {
    }
  }
}
