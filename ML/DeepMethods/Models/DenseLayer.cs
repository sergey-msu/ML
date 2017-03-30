using System;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// General neural network one-dimensional fully-connected layer
  /// </summary>
  public class DenseLayer : ConvolutionalLayer
  {
    public DenseLayer(int inputDim,
                      int outputDim,
                      bool isTraining = false)
      : base(inputDim,
             1,
             outputDim,
             windowSize: 1,
             stride: 1,
             padding: 0,
             isTraining: isTraining)
    {
    }
  }
}
