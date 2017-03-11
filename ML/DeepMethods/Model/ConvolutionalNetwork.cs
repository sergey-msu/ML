using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core.ComputingNetworks;
using ML.Core;

namespace ML.DeepMethods.Model
{
  /// <summary>
  /// Represents feedforward convolutional neural network: set of convolutional layers
  /// with convolutional neuron nodes and shared weights
  /// </summary>
  public class ConvolutionalNetwork : NetworkNode<double[,], ConvolutionalLayer, FeatureMap>
  {
    private int m_InputSize;

    public ConvolutionalNetwork(int inputDim, int inputSize)
      : base(inputDim)
    {
      if (inputSize <= 0)
        throw new MLException("ConvolutionNeuron.ctor(inputSize<=0)");

      m_InputSize = inputSize;
    }


    /// <summary>
    /// Size of square input matrix
    /// </summary>
    public int InputSize { get { return m_InputSize; } }
  }
}
