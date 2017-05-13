using System;
using ML.Contracts;

namespace ML.DeepMethods.Regularization
{
  public class L1Regularizator : IRegularizator
  {
    private double m_Coeff;

    public L1Regularizator(double coeff)
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
          result += Math.Abs(w[j]);
      }

      return m_Coeff*result;
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
          g[j] += m_Coeff*(w[j]>0 ? 1 : -1);
      }
    }
  }
}
