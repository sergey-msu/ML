using System;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Adaptive gradient (Adagrad) optimizer:
  ///
  /// G[t+1] = G[t] + DL(w[t])*DL(w[t])
  /// w[t+1] = w[t] + lr/sqrt(G[t] + eps) * DL(w[t])
  ///
  /// </summary>
  public class AdagradOptimizer : OptimizerBase
  {
    public const double DFT_EPSILON = 1.0E-8D;

    private double m_Epsilon;
    private double[][] m_G;

    public AdagradOptimizer(double epsilon = DFT_EPSILON)
    {
      if (epsilon<=0)
        throw new MLException("Epsilon must be positive");

      m_Epsilon = epsilon;
    }

    public double Epsilon { get { return m_Epsilon; } }


    protected override void DoPush(double[][] weights, double[][] gradient, double learningRate)
    {
      var len = weights.Length;
      var step2 = 0.0D;

      if (m_G==null)
      {
        m_G = new double[len][];
        for (int i=0; i<len; i++)
        {
          var layerWeights = weights[i];
          if (layerWeights==null) continue;

          m_G[i] = new double[layerWeights.Length];
        }
      }

      for (int i=0; i<len; i++)
      {
        var layerWeights = weights[i];
        if (layerWeights==null) continue;

        var wlen = layerWeights.Length;
        var layerGradient = gradient[i];
        var gi = m_G[i];

        for (int j=0; j<wlen; j++)
        {
          var g  = layerGradient[j];
          gi[j] += g*g;
          var dw = -learningRate/Math.Sqrt(gi[j] + m_Epsilon) * g;
          step2 += dw*dw;

          layerWeights[j] += dw;
        }

        Array.Clear(layerGradient, 0, layerGradient.Length);
      }

      m_Step2 = step2;
    }
  }
}
