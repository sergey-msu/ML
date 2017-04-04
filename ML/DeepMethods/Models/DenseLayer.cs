using System;
using ML.Contracts;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// General neural network one-dimensional fully-connected layer
  /// </summary>
  public class DenseLayer : ConvolutionalLayer
  {
    public DenseLayer(int outputDim, IActivationFunction activation = null)
      : base(outputDim,
             windowSize: 1,
             stride: 1,
             padding: 0,
             activation: activation)
    {
    }
  }
}
