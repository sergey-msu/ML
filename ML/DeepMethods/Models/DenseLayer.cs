using System;
using ML.Contracts;
using ML.Core;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// General neural network one-dimensional fully-connected layer
  /// </summary>
  public class DenseLayer : ConvLayer
  {
    public DenseLayer(int outputDim, IActivationFunction activation = null)
      : base(outputDim,
             windowSize: 1,
             stride: 1,
             padding: 0,
             activation: activation)
    {
    }


    protected override void BuildShape()
    {
      if (m_InputHeight != 1 || m_InputWidth != 1)
        throw new MLException("Cannot apply dense layer after non-dense/flatten one. Use flatten layer first");

      base.BuildShape();
    }

    protected override void DoCalculate(double[][,] input, double[][,] result)
    {
      var plen = m_InputDepth + 1;

      for (int q=0; q<m_OutputDepth; q++)
      {
        var net = m_Weights[(q+1)*plen-1]; // Bias(q)
        for (int p=0; p<m_InputDepth; p++)
          net += input[p][0, 0] * m_Weights[p + q*plen]; // Kernel(q, p, 0, 0)

        result[q][0, 0] = (m_ActivationFunction != null) ? m_ActivationFunction.Value(net) : net;
      }
    }

    /// <summary>
    /// Backpropagate "errors" to previous layer for future use
    /// </summary>
    protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] prevValues, double[][,] prevError, double[][,] errors)
    {
      if (prevLayer==null)
        throw new MLException("Prev layer is null");

      var plen = m_InputDepth + 1;

      for (int p=0; p<m_InputDepth; p++)
      {
        var g = 0.0D;
        for (int q=0; q<m_OutputDepth; q++)
          g += errors[q][0, 0] * m_Weights[p + q*plen]; // Kernel(q, p, 0, 0)

        var value = prevValues[p][0, 0];
        var deriv = (prevLayer.ActivationFunction != null) ? prevLayer.ActivationFunction.DerivativeFromValue(value) : 1;
        prevError[p][0, 0] = g * deriv;
      }
    }

    protected override void DoSetLayerGradient(double[][,] prevValues, double[][,] errors, double[] gradient, bool isDelta)
    {
      var plen = m_InputDepth + 1;
      int idx;

      // weight updates
      for (int q=0; q<m_OutputDepth; q++)
      {
        for (int p=0; p<m_InputDepth;   p++)
        for (int y=0; y<m_WindowHeight; y++)
        for (int x=0; x<m_WindowWidth;  x++)
        {
          idx = p + q*plen;
          var dw = errors[q][0, 0] * prevValues[p][y, x];
          if (isDelta) gradient[idx] += dw; // Gradient(q, p, i, j)
          else gradient[idx] = dw;
        }

        // bias updates
        idx = (q+1)*m_FeatureMapParamCount-1;
        var db = errors[q][0, 0];
        if (isDelta) gradient[idx] += db;  // BiasGrad(q)
        else gradient[idx] = db;
      }
    }

  }
}