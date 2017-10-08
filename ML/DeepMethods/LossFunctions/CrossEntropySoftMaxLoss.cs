using System;
using ML.Contracts;
using ML.Core;
using ML.Utils;

namespace ML.DeepMethods.LossFunctions
{
  public class CrossEntropySoftMaxLoss : ILossFunction
  {
    public double Value(double[] actual, double[] expected)
    {
      var res = 0.0D;
      var sa = 0.0D;
      var se = 0.0D;
      var len = actual.Length;
      for (int i=0; i<len; i++)
      {
        sa += actual[i];
        se += expected[i];
        res += expected[i] * Math.Log(actual[i]);
      }

      var result = (se*Math.Log(sa) - res) * MathConsts.ENTROPY_COEFF;
      if (double.IsNaN(result))
        throw new MLException(string.Format("Argument of entropy is out of range. Make sure that activation function's output is (0, 1). Values: se={0}, sa={1}", se, sa));

      return result;
    }

    public double Derivative(int idx, double[] actual, double[] expected)
    {
      var sa = 0.0D;
      var se = 0.0D;
      var len = actual.Length;
      for (int i=0; i<len; i++)
      {
        sa += actual[i];
        se += expected[i];
      }

      var result = (se/sa - expected[idx]/actual[idx]) * MathConsts.ENTROPY_COEFF;
      if (double.IsNaN(result))
        throw new MLException(string.Format("Argument of entropy is out of range. Make sure that activation function's output is (0, 1). Values: idx={0}, actual={1}, expected={2}, se={3}, sa={4}", idx, actual[idx], expected[idx], se, sa));

      return result;
    }
  }
}
