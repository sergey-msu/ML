using System;
using ML.Contracts;
using ML.Core;

namespace ML.DeepMethods.LossFunctions
{
  public class LpLoss : ILossFunction
  {
    private readonly double m_P;

    public LpLoss(double p)
    {
      if (p<1)
        throw new MLException("P must be greater than or equals 1");

      m_P = p;
    }

    public double Value(double[] actual, double[] expected)
    {
      var summ = 0.0D;
      var len = actual.Length;
      for (int i=0; i<len; i++)
      {
        var diff = actual[i] - expected[i];
        summ += Math.Pow(Math.Abs(diff), m_P);
      }

      return summ/m_P;
    }

    public double Derivative(int idx, double[] actual, double[] expected)
    {
      var diff = actual[idx] - expected[idx];
      return Math.Pow(Math.Abs(diff), m_P-2)*diff;
    }
  }
}
