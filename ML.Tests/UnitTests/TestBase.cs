using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ML.Tests.UnitTests
{
  [TestClass]
  public class TestBase
  {
    private Random m_Random = new Random(0);

    public const double EPS = 0.0000001D;
    public const double EPS_ROUGH = 0.00001D;
    public const double GRAD_EPS  = 0.00001D;
    public const double GRAD_STEP = 0.00001D;

    private static TestContext m_Context;

    public static TestContext Context { get { return m_Context; } }

    [ClassInitialize]
    public static void BaseClassInit(TestContext context)
    {
      m_Context = context;
    }

    protected void AssertDerivative(Func<double, double> f, double x, double actual)
    {
      var expected = (f(x+GRAD_STEP) - f(x-GRAD_STEP)) / (2*GRAD_STEP);

      if (Math.Abs(expected) <= double.Epsilon)
        Assert.IsTrue(Math.Abs(actual) <= double.Epsilon);
      else
        Assert.IsTrue(Math.Abs(actual-expected)/expected < GRAD_EPS);
    }

    protected void ResetRandom(int seed)
    {
      m_Random = new Random(seed);
    }

    protected double[][,] RandomPoint(int l1, int l2, int l3)
    {
      var result = new double[l1][,];

      for (int i=0; i<l1; i++)
      {
        var s = new double[l2, l3];
        for (int j=0; j<s.GetLength(0); j++)
        for (int k=0; k<s.GetLength(1); k++)
          s[j,k] = m_Random.NextDouble();
        result[i] = s;
      }

      return result;
    }

    protected void Randomize(double[][] weights, double min, double max)
    {
      for (int i=0; i<weights.Length; i++)
      {
        var w = weights[i];
        if (w==null) continue;
        for (int j=0; j<w.Length; j++)
          w[j] = m_Random.NextDouble()*(max-min) + min;
      }
    }
  }
}
