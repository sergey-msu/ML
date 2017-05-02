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

    protected override void DoCalculate(double[][,] input, double[][,] result)
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
          var idx = x + y*m_WindowWidth + p*m_KernelParamCount + q*m_FeatureMapParamCount;
          net += input[p][y, x] * m_Weights[idx]; // Kernel(q, p, y, x)
        }

        result[q][0, 0] = (m_ActivationFunction != null) ? m_ActivationFunction.Value(net) : net;
      }
    }

    /// <summary>
    /// Backpropagate "errors" to previous layer for future use
    /// </summary>
    protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] prevValues, double[][,] prevErrors, double[][,] errors)
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
          var idx = x + y*m_WindowWidth + p*m_KernelParamCount + q*m_FeatureMapParamCount;
          g += errors[q][0, 0] * m_Weights[idx]; // Kernel(q, p, y, x)
        }

        var value = prevValues[p][y, x];
        var deriv = (prevLayer.ActivationFunction != null) ? prevLayer.ActivationFunction.DerivativeFromValue(value) : 1;
        prevErrors[p][y, x] = g * deriv;
      }
    }

    protected override void DoSetLayerGradient(double[][,] prevValues, double[][,] errors, double[] gradient, bool isDelta)
    {
      int idx;

      // weight updates
      for (int q=0; q<m_OutputDepth; q++)
      {
        for (int p=0; p<m_InputDepth;   p++)
        for (int y=0; y<m_WindowHeight; y++)
        for (int x=0; x<m_WindowWidth;  x++)
        {
          idx = x + y*m_WindowWidth + p*m_KernelParamCount + q*m_FeatureMapParamCount;
          var dw = errors[q][0, 0] * prevValues[p][y, x];
          if (isDelta) gradient[idx] += dw; // Gradient(q, p, y, x)
          else gradient[idx] = dw;
        }

        // bias updates
        idx = (q+1)*m_FeatureMapParamCount-1;
        var db = errors[q][0, 0];
        if (isDelta) gradient[idx] += db; // BiasGrad(q)
        else gradient[idx] = db;
      }
    }

  }
}
