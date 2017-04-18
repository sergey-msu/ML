using System;
using ML.Contracts;
using ML.Core.Mathematics;

namespace ML.DeepMethods.LossFunctions
{
  public class NeuralMethods : ILossFunction
  {
    public double Value(double[] actual, double[] expected)
    {
      var res = 0.0D;
      var len = actual.Length;
      for (int i=0; i<len; i++)
      {
        res += expected[i] * Math.Log(actual[i]);
      }

      return -res * MathUtils.ENTROPY_COEFF;
    }

    public double Derivative(int idx, double[] actual, double[] expected)
    {
      return -MathUtils.ENTROPY_COEFF * expected[idx] / actual[idx];
    }
  }
}
