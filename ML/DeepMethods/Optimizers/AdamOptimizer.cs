using System;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Adaptive moment estimation (Adam) optimizer.
  /// (see https://arxiv.org/pdf/1412.6980.pdf)
  /// </summary>
  public class AdamOptimizer : OptimizerBase
  {
    private double[][] m_M;
    private double[][] m_V;
    private double m_Beta1t;
    private double m_Beta2t;

    private double m_Beta1;
    private double m_Beta2;
    private double m_Epsilon;

    public AdamOptimizer(double beta1, double beta2, double epsilon)
    {
      if (beta1<0 || beta1>=1)
        throw new MLException("Beta_1 must be within [0,1) interval");
      if (beta2<0 || beta2>=1)
        throw new MLException("Beta_2 must be within [0,1) interval");
      if (epsilon < 0)
        throw new MLException("Epsilon must be positive");

      m_Beta1   = beta1;
      m_Beta2   = beta2;
      m_Beta1t  = beta1;
      m_Beta2t  = beta2;
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
        m_V = new double[len][];

        for (int i=0; i<len; i++)
        {
          var layerWeights = weights[i];
          if (layerWeights==null) continue;

          m_M[i] = new double[layerWeights.Length];
          m_V[i] = new double[layerWeights.Length];
        }
      }

      for (int i=0; i<len; i++)
      {
        var layerWeights = weights[i];
        if (layerWeights==null) continue;

        var wlen = layerWeights.Length;
        var layerGradient = gradient[i];
        var mi = m_M[i];
        var vi = m_V[i];

        for (int j=0; j<wlen; j++)
        {
          var g  = layerGradient[j];
          mi[j] = m_Beta1*mi[j] + (1-m_Beta1)*g;
          vi[j] = m_Beta2*vi[j] + (1-m_Beta2)*g*g;
          var mih = mi[j]/(1 - m_Beta1t);
          var vih = vi[j]/(1 - m_Beta2t);

          var dw = -learningRate * mih / (Math.Sqrt(vih)+m_Epsilon);
          step2 += dw*dw;

          layerWeights[j] += dw;
        }

        Array.Clear(layerGradient, 0, layerGradient.Length);
      }

      m_Beta1t *= m_Beta1;
      m_Beta2t *= m_Beta2;
      m_Step2 = step2;
    }
  }
}
