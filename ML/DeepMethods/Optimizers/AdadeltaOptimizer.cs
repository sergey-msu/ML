using System;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Adaptive delta gradient (Adadelta) optimizer
  /// (see https://arxiv.org/pdf/1212.5701.pdf)
  /// </summary>
  public class AdadeltaOptimizer : OptimizerBase
  {
    public const double DFT_GAMMA     = 0.9D;
    public const double DFT_EPSILON   = 1.0E-9D;
    public const bool   DFT_USE_LRATE = false;

    private double m_Epsilon;
    private double m_Gamma;
    private bool   m_UseLearningRate;
    private double[][] m_E;
    private double[][] m_ED;

    public AdadeltaOptimizer(double gamma = DFT_GAMMA, double epsilon = DFT_EPSILON, bool useLearningRate = DFT_USE_LRATE)
    {
      if (epsilon<=0)
        throw new MLException("Epsilon must be positive");
      if (gamma<0 || gamma>1)
        throw new MLException("Gamma must be within [0,1] interval");

      m_Epsilon = epsilon;
      m_Gamma = gamma;
      m_UseLearningRate = useLearningRate;
    }

    public double Epsilon { get { return m_Epsilon; } }
    public double Gamma   { get { return m_Gamma; } }
    public bool   UseLearningRate { get { return m_UseLearningRate; } }


    protected override void DoPush(double[][] weights, double[][] gradient, double learningRate)
    {
      var len = weights.Length;
      var step2 = 0.0D;

      if (m_E==null)
      {
        m_E  = new double[len][];
        m_ED = new double[len][];

        for (int i=0; i<len; i++)
        {
          var layerWeights = weights[i];
          if (layerWeights==null) continue;

          m_E[i]  = new double[layerWeights.Length];
          m_ED[i] = new double[layerWeights.Length];
        }
      }

      for (int i=0; i<len; i++)
      {
        var layerWeights = weights[i];
        if (layerWeights==null) continue;

        var wlen = layerWeights.Length;
        var layerGradient = gradient[i];
        var ei = m_E[i];
        var edi = m_ED[i];

        for (int j=0; j<wlen; j++)
        {
          var g   = layerGradient[j];
          ei[j]   = m_Gamma*ei[j] + (1-m_Gamma)*g*g;
          var dw  = -Math.Sqrt((edi[j]+m_Epsilon)/(ei[j]+m_Epsilon)) * g;
          var dw2 = dw*dw;
          edi[j]  = m_Gamma*edi[j] + (1-m_Gamma)*dw2;

          if (m_UseLearningRate)
          {
            dw *= learningRate;
            dw2 *= (learningRate*learningRate);
          }

          step2 += dw2;

          layerWeights[j] += dw;
        }

        Array.Clear(layerGradient, 0, layerGradient.Length);
      }

      m_Step2 = step2;
    }
  }
}
