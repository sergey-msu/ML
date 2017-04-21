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
  public class NopeOptimizer : OptimizerBase
  {
    public NopeOptimizer()
    {
    }


    /// <summary>
    /// Push current gradient vector to optimizer
    /// </summary>
    public override void Push(double[][] gradient, double learningRate)
    {
      var len = m_Weights.Length;
      var step2 = 0.0D;

      for (int i=0; i<len; i++)
      {
        var layerWeights = m_Weights[i];
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
