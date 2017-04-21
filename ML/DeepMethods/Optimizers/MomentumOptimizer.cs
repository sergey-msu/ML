using System;
using ML.Contracts;
using ML.Core;

namespace ML.DeepMethods.Optimizers
{
  /// <summary>
  /// Simple momentum optimizer:
  ///
  /// dw[t+1] = µ[t]w[t] − lr[t]DL(w[t])
  ///  w[t+1] = w[t] + dw[t+1]
  ///
  /// </summary>
  public class MomentumOptimizer : OptimizerBase
  {
    private double m_Mu;
    private double[][] m_UpdateHistory;

    public MomentumOptimizer(double mu)
    {
      if (mu<0 || mu>1)
        throw new MLException("Mu parameter must be in [0,1] interval");

      m_Mu = mu;
    }



    public override void Push(double[][] gradient, double learningRate)
    {
      var len = m_Weights.Length;
      var step2 = 0.0D;

      if (m_UpdateHistory==null)
      {
        m_UpdateHistory = new double[len][];
        for (int i=0; i<len; i++)
        {
          var layerWeights = m_Weights[i];
          if (layerWeights==null) continue;

          m_UpdateHistory[i] = new double[layerWeights.Length];
        }
      }

      for (int i=0; i<len; i++)
      {
        var layerWeights = m_Weights[i];
        if (layerWeights==null) continue;

        var wlen = layerWeights.Length;
        var layerGradient = gradient[i];
        var history = m_UpdateHistory[i];

        for (int j=0; j<wlen; j++)
        {
          var dw = m_Mu*history[j] - learningRate*layerGradient[j];
          history[j] = dw;
          layerWeights[j] += dw;
          step2 += dw*dw;
        }
      }
    }

  }
}
