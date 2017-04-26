using System;
using ML.Contracts;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Standart SGD optimizer - does no actual optimization, simply applies update vector as is:
  /// dw = - l*dL/dw
  /// w := w + dw
  /// </summary>
  public class SGDOptimizer : OptimizerBase
  {
    public SGDOptimizer()
    {
    }

    /// <summary>
    /// Push current gradient vector to optimizer
    /// </summary>
    protected override void DoPush(double[][] weights, double[][] gradient, double learningRate)
    {
      var len = weights.Length;
      var step2 = 0.0D;

      for (int i=0; i<len; i++)
      {
        var layerWeights = weights[i];
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
