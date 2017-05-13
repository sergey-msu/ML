using System;
using ML.Contracts;

namespace ML.DeepMethods.Regularization
{
  public class L2Regularizator : IRegularizator
  {
    private double m_Coeff;

    public L2Regularizator(double coeff)
    {
      m_Coeff = coeff;
    }

    public double Coeff { get { return m_Coeff; } }


    public double Value(double[][] weights)
    {
      var result = 0.0D;

      var len = weights.Length;
      for (int i=0; i<len; i++)
      {
        var w = weights[i];
        if (w==null) continue;
        for (int j=0; j<w.Length; j++)
          result += (w[j]*w[j]);
      }

      return m_Coeff*result;
    }

    public double Derivative(int idx1, int idx2, double[][] weights)
    {
      return 2*m_Coeff*weights[idx1][idx2];
    }

    public void Apply(double[][] gradients, double[][] weights)
    {
      var len = weights.Length;
      for (int i=0; i<len; i++)
      {
        var w = weights[i];
        if (w==null) continue;
        var g = gradients[i];
        for (int j=0; j<w.Length; j++)
          g[j] += (m_Coeff*w[j]*w[j]);
      }
    }
  }
}
