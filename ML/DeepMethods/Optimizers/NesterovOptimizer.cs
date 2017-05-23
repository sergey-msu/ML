using System;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Nesterov momentum optimizer:
  ///
  ///  v[t+1] =  mu*v[t] - lr*DL[t]
  /// dw[t+1] = -mu*v[t] + (1 + mu)*v[t+1]
  ///  w[t+1] =  w[t] + dw[t+1]
  ///
  ///  (see http://cs231n.github.io/neural-networks-3)
  /// </summary>
  public class NesterovOptimizer : OptimizerBase
  {
    public const double DFT_MU = 0.9D;

    private double m_Mu;
    private double[][] m_V;

    public NesterovOptimizer(double mu = DFT_MU)
    {
      if (mu<0 || mu>1)
        throw new MLException("Mu parameter must be in [0,1] interval");

      m_Mu = mu;
    }

    public double Mu { get { return m_Mu; } }


    protected override void DoPush(double[][] weights, double[][] gradient, double learningRate)
    {
      var len = weights.Length;
      var step2 = 0.0D;

      if (m_V==null)
      {
        m_V = new double[len][];
        for (int i=0; i<len; i++)
        {
          var layerWeights = weights[i];
          if (layerWeights==null) continue;

          m_V[i] = new double[layerWeights.Length];
        }
      }

      for (int i=0; i<len; i++)
      {
        var layerWeights = weights[i];
        if (layerWeights==null) continue;

        var wlen = layerWeights.Length;
        var layerGradient = gradient[i];
        var vi = m_V[i];

        for (int j=0; j<wlen; j++)
        {
          var ov = vi[j];
          var nv = m_Mu*ov - learningRate*layerGradient[j];
          vi[j]  = nv;
          var dw = -m_Mu*ov + (1+m_Mu)*nv;
          layerWeights[j] += dw;
          step2 += dw*dw;
        }

        Array.Clear(layerGradient, 0, layerGradient.Length);
      }

      m_Step2 = step2;
    }

  }
}
