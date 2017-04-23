using System;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Adaptive delta gradient (Adadelta) optimizer:
  ///
  /// E[t+1] = gam*E[t] + (1-gam)*DL(w[t])*DL(w[t])
  /// w[t+1] = w[t] - lr/sqrt(E[t] + eps) * DL(w[t])
  ///
  /// </summary>
  public class AdadeltaOptimizer : OptimizerBase
  {
    private double m_Epsilon;
    private double m_Gamma;
    private double[][] m_E;
    private double[][] m_ED;

    public AdadeltaOptimizer(double epsilon, double gamma)
    {
      if (epsilon<=0)
        throw new MLException("Epsilon must be positive");
      if (gamma<0 || gamma>1)
        throw new MLException("Gamma must be within [0,1] interval");

      m_Epsilon = epsilon;
      m_Gamma = gamma;
    }

    public double Epsilon { get { return m_Epsilon; } }


    public override void Push(double[][] gradient, double learningRate)
    {
      var len = m_Weights.Length;
      var step2 = 0.0D;

      if (m_E==null)
      {
        m_E = new double[len][];
        m_ED = new double[len][];
        for (int i=0; i<len; i++)
        {
          var layerWeights = m_Weights[i];
          if (layerWeights==null) continue;

          m_E[i] = new double[layerWeights.Length];
          m_ED[i] = new double[layerWeights.Length];
        }
      }

      for (int i=0; i<len; i++)
      {
        var layerWeights = m_Weights[i];
        if (layerWeights==null) continue;

        var wlen = layerWeights.Length;
        var layerGradient = gradient[i];
        var ei = m_E[i];

        for (int j=0; j<wlen; j++)
        {
          var g  = layerGradient[j];
          var g2 = g*g;
          ei[j] = m_Gamma*ei[j] + (1-m_Gamma)*g2;

          var dw = -learningRate*layerGradient[j] / Math.Sqrt(ei[j] + m_Epsilon);
          layerWeights[j] += dw;
          step2 += dw*dw;
        }

        Array.Clear(layerGradient, 0, layerGradient.Length);
      }

      m_Step2 = step2;
    }
  }
}
