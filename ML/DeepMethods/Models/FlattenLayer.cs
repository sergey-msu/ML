using System;
using ML.Contracts;
using ML.Core;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Flattening layer that transform multidimensional array data into flat one-dimensional fully-connected layer
  /// </summary>
  public class FlattenLayer : ConvLayer
  {
    public FlattenLayer(int outputDim, IActivationFunction activation = null)
      : base(outputDim,
             windowSize: 1, // will be overridden with input size when building the layer
             stride: 1,
             padding: 0,
             activation: activation)
    {
    }


    protected override void BuildShape()
    {
      m_WindowHeight = m_InputHeight;
      m_WindowWidth  = m_InputWidth;

      base.BuildShape();
    }

    protected override double[][,] DoCalculate(double[][,] input)
    {
      // output fm-s
      for (int q=0; q<m_OutputDepth; q++)
      {
        var net = m_Weights[(q+1)*m_FeatureMapParamCount-1]; // Bias(q)

        // flatten window
        for (int y=0; y<m_WindowHeight; y++)
        for (int x=0; x<m_WindowWidth;  x++)
        for (int p=0; p<m_InputDepth;   p++) // inner product in p-depth (over input channel's neuron at fixed position)
        {
          net += input[p][y, x] *
                 m_Weights[x + y*m_WindowWidth + p*m_KernelParamCount + q*m_FeatureMapParamCount]; // Kernel(q, p, y, x)
        }

        m_Value[q][0, 0] = (m_ActivationFunction != null) ? m_ActivationFunction.Value(net) : net;
      }

      return m_Value;
    }

    /// <summary>
    /// Backpropagate "errors" to previous layer for future use
    /// </summary>
    protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] error, double[][,] prevError)
    {
      if (prevLayer==null)
        throw new MLException("Prev layer is null");

      for (int p=0; p<m_InputDepth;   p++)
      for (int y=0; y<m_WindowHeight; y++)
      for (int x=0; x<m_WindowWidth;  x++)
      {
        var g = 0.0D;

        for (int q=0; q<m_OutputDepth; q++)
        {
          g += error[q][0, 0] *
               m_Weights[x + y*m_WindowWidth + p*m_KernelParamCount + q*m_FeatureMapParamCount]; // Kernel(q, p, y, x)
        }

        prevError[p][y, x] = g * prevLayer.Derivative(p, y, x);
      }
    }

    protected override void DoSetLayerGradient(DeepLayerBase prevLayer, double[][,] errors, double[] layerGradient)
    {
      // weight updates
      for (int q=0; q<m_OutputDepth; q++)
      {
        for (int p=0; p<m_InputDepth;   p++)
        for (int y=0; y<m_WindowHeight; y++)
        for (int x=0; x<m_WindowWidth;  x++)
        {
          layerGradient[x + y*m_WindowWidth + p*m_KernelParamCount + q*m_FeatureMapParamCount] += errors[q][0, 0] * prevLayer.Value(p, y, x); // Gradient(q, p, y, x)
        }

        // bias updates
        var db = errors[q][0, 0];
        layerGradient[(q+1)*m_FeatureMapParamCount-1] += db;  // BiasGrad(q)
      }
    }

  }
}
