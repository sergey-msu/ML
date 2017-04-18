using ML.Contracts;

namespace ML.DeepMethods.LossFunctions
{
  public class EuclideanLoss : ILossFunction
  {
    public double Value(double[] actual, double[] expected)
    {
      var summ = 0.0D;
      var len = actual.Length;
      for (int i=0; i<len; i++)
      {
        var diff = actual[i] - expected[i];
        summ += diff*diff;
      }

      return summ/2;
    }

    public double Derivative(int idx, double[] actual, double[] expected)
    {
      return actual[idx] - expected[idx];
    }
  }
}
