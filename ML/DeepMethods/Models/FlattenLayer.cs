using System;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Flattening layer that transform multidimensional array data into flat one-dimensional fully-connected layer
  /// </summary>
  public class FlattenLayer : ConvolutionalLayer
  {
    public FlattenLayer(int outputDim)
      : base(outputDim,
             windowSize: 1, // to be overridden with input size on build
             stride: 1,
             padding: 0)
    {
    }

    public override void DoBuild()
    {
      m_WindowSize = m_InputSize;

      base.DoBuild();
    }
  }
}
