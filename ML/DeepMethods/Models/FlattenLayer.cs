using System;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Flattening layer that transform multidimensional array data into flat one-dimendional fully-connected layer
  /// </summary>
  public class FlattenLayer : ConvolutionalLayer
  {
    public FlattenLayer(int inputDepth,
                        int inputSize,
                        int outputDim,
                        bool isTraining = false)
      : base(inputDepth,
             inputSize,
             outputDim,
             inputSize,
             stride: 1,
             padding: 0,
             isTraining: isTraining)
    {
    }
  }
}
