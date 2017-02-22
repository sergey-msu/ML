using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core.ComputingNetworks;

namespace ML.NeuralMethods.Contracts
{
  public interface INeuralNetwork : IComputingNode<double[], object>
  {
    IFunction ActivationFunction { get; set; }
  }
}
