using System;
using ML.Contracts;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Trivial optimizer - does no actual optimization, simply applies update vector as is:
  /// dw = - l*dL/dw
  /// w := w + dw
  /// </summary>
  public class NopeOptimizer : IOptimizer
  {
    private double[][] m_Weights;
    private double m_Step2;


    public NopeOptimizer()
    {
    }

    /// <summary>
    /// Last weight vector step value (squared)
    /// </summary>
    public double Step2 { get { return m_Step2; } }

    /// <summary>
    /// Set source weight vector
    /// </summary>
    public void Init(double[][] weights)
    {
      if (weights==null)
        throw new MLException("Weights can not be null");

      m_Weights = weights;
    }

    /// <summary>
    /// Push current gradient vector to optimizer
    /// </summary>
    public void Push(double[][] gradient, double learningRate)
    {
      var len = m_Weights.Length;
      var step2 = 0.0D;

      for (int i=len-1; i>=0; i--)
      {
        var layerWeights  = m_Weights[i];
        if (layerWeights==null) continue;

        var layerGradient = gradient[i];
        var wlen = layerWeights.Length;
        for (int j=0; j<wlen; j++)
        {
          var dw = -learningRate * layerGradient[j];
          layerWeights[j] += dw;
          step2 += dw*dw;
        }

        Array.Clear(layerGradient, 0, layerGradient.Length);
      }

      m_Step2 = step2;
    }
  }
}
