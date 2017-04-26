using System;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Adaptive moment estimation with l_infty norm (Adamax) optimizer.
  /// (see https://arxiv.org/pdf/1412.6980.pdf)
  /// </summary>
  public class AdamaxOptimizer : OptimizerBase
  {
    public const double DFT_BETA1   = 0.9D;
    public const double DFT_BETA2   = 0.999D;
    public const double DFT_EPSILON = 1.0E-8D;

    private double[][] m_M;
    private double[][] m_U;
    private double m_Beta1t;

    private double m_Beta1;
    private double m_Beta2;
    private double m_Epsilon;

    public AdamaxOptimizer(double beta1 = DFT_BETA1, double beta2 = DFT_BETA2, double epsilon = DFT_EPSILON)
    {
      if (beta1<0 || beta1>=1)
        throw new MLException("Beta_1 must be within [0,1) interval");
      if (beta2<0 || beta2>=1)
        throw new MLException("Beta_2 must be within [0,1) interval");
      if (epsilon < 0)
        throw new MLException("Epsilon must be positive");

      m_Beta1   = beta1;
      m_Beta1t  = beta1;
      m_Beta2   = beta2;
      m_Epsilon = epsilon;
    }

    public double Beta1   { get { return m_Beta1; } }
    public double Beta2   { get { return m_Beta2; } }
    public double Epsilon { get { return m_Epsilon; } }


    protected override void DoPush(double[][] weights, double[][] gradient, double learningRate)
    {
      var len = weights.Length;
      var step2 = 0.0D;

      if (m_M==null)
      {
        m_M = new double[len][];
        m_U = new double[len][];

        for (int i=0; i<len; i++)
        {
          var layerWeights = weights[i];
          if (layerWeights==null) continue;

          m_M[i] = new double[layerWeights.Length];
          m_U[i] = new double[layerWeights.Length];
        }
      }

      for (int i=0; i<len; i++)
      {
        var layerWeights = weights[i];
        if (layerWeights==null) continue;

        var wlen = layerWeights.Length;
        var layerGradient = gradient[i];
        var mi = m_M[i];
        var ui = m_U[i];

        for (int j=0; j<wlen; j++)
        {
          var g  = layerGradient[j];
          mi[j] = m_Beta1*mi[j] + (1-m_Beta1)*g;
          ui[j] = Math.Max(m_Beta2*ui[j], Math.Abs(g));
          var mih = mi[j]/(1 - m_Beta1t);

          var dw = -learningRate * mih/(ui[j]+m_Epsilon);
          step2 += dw*dw;

          layerWeights[j] += dw;
        }

        Array.Clear(layerGradient, 0, layerGradient.Length);
      }

      m_Beta1t *= m_Beta1;
      m_Step2 = step2;
    }
  }
}
