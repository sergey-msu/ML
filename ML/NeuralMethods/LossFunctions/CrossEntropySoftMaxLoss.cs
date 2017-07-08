using System;
using ML.Contracts;
using ML.Core.Mathematics;
using ML.Core;
using ML.Utils;

namespace ML.NeuralMethods.LossFunctions
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

      return (se*Math.Log(sa) - res) * GeneralUtils.ENTROPY_COEFF;
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

      var result = (se/sa - expected[idx]/actual[idx]) * GeneralUtils.ENTROPY_COEFF;
      if (double.IsNaN(result))
        throw new MLException("Argument of entropy is out of range. Make sure that activation function's output is (0, 1)");

      return result;
    }
  }
}
