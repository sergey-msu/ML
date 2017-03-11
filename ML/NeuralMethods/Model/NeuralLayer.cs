using System;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.NeuralMethods.Model
{
  /// <summary>
  /// Represents artificial neural layer as a list of neurons
  /// </summary>
  public class NeuralLayer : LayerNode<double, Neuron>
  {
    #region .ctor

    public NeuralLayer(int inputDim) : base(inputDim)
    {
    }

    #endregion
  }
}
