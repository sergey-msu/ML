using System;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Root mean square propagation (RMSprop) optimizer:
  ///
  /// E[t+1] = gam*E[t] + (1-gam)*DL(w[t])*DL(w[t])
  /// w[t+1] = w[t] - lr/sqrt(E[t] + eps) * DL(w[t])
  ///
  /// </summary>
  public class RMSPropOptimizer : OptimizerBase
  {
    public const double DFT_GAMMA   = 0.9D;
    public const double DFT_EPSILON = 1.0E-8D;

    private double m_Epsilon;
    private double m_Gamma;
    private double[][] m_E;

    public RMSPropOptimizer(double gamma = DFT_GAMMA, double epsilon = DFT_EPSILON)
    {
      if (epsilon<=0)
        throw new MLException("Epsilon must be positive");
      if (gamma<0 || gamma>1)
        throw new MLException("Gamma must be within [0,1] interval");

      m_Epsilon = epsilon;
      m_Gamma = gamma;
    }

    public double Epsilon { get { return m_Epsilon; } }
    public double Gamma { get { return m_Gamma; } }


    protected override void DoPush(double[][] weights, double[][] gradient, double learningRate)
    {
      var len = weights.Length;
      var step2 = 0.0D;

      if (m_E==null)
      {
        m_E = new double[len][];
        for (int i=0; i<len; i++)
        {
          var layerWeights = weights[i];
          if (layerWeights==null) continue;

          m_E[i] = new double[layerWeights.Length];
        }
      }

      for (int i=0; i<len; i++)
      {
        var layerWeights = weights[i];
        if (layerWeights==null) continue;

        var wlen = layerWeights.Length;
        var layerGradient = gradient[i];
        var ei = m_E[i];

        for (int j=0; j<wlen; j++)
        {
          var g  = layerGradient[j];
          ei[j] = m_Gamma*ei[j] + (1-m_Gamma)*g*g;
          var dw = -learningRate/Math.Sqrt(ei[j] + m_Epsilon) * g;
          step2 += dw*dw;

          layerWeights[j] += dw;
        }

        Array.Clear(layerGradient, 0, layerGradient.Length);
      }

      m_Step2 = step2;
    }
  }
}
