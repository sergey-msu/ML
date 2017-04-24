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
    private double m_Beta1;
    private double m_Beta2;

    public AdamOptimizer(double beta1, double beta2)
    {
      if (beta1<0 || beta1>1)
        throw new MLException("Beta_1 must be within [0,1] interval");
      if (beta2<0 || beta2>1)
        throw new MLException("Beta_2 must be within [0,1] interval");

      m_Beta1 = beta1;
      m_Beta2 = beta2;
    }

    public double Beta1 { get { return m_Beta1; } }
    public double Beta2 { get { return m_Beta2; } }


    public override void Push(double[][] gradient, double learningRate)
    {
      var len = m_Weights.Length;
      var step2 = 0.0D;

      //if (m_G==null)
      //{
      //  m_G = new double[len][];
      //  for (int i=0; i<len; i++)
      //  {
      //    var layerWeights = m_Weights[i];
      //    if (layerWeights==null) continue;
      //
      //    m_G[i] = new double[layerWeights.Length];
      //  }
      //}
      //
      //for (int i=0; i<len; i++)
      //{
      //  var layerWeights = m_Weights[i];
      //  if (layerWeights==null) continue;
      //
      //  var wlen = layerWeights.Length;
      //  var layerGradient = gradient[i];
      //  var gi = m_G[i];
      //
      //  for (int j=0; j<wlen; j++)
      //  {
      //    var g  = layerGradient[j];
      //    var g2 = g*g;
      //    gi[j] += g2;
      //    var dw = -learningRate/Math.Sqrt(gi[j] + m_Epsilon) * g;
      //    step2 += dw*dw;
      //
      //    layerWeights[j] += dw;
      //  }
      //
      //  Array.Clear(layerGradient, 0, layerGradient.Length);
      //}

      m_Step2 = step2;
    }
  }
}
