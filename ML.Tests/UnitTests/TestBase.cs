using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ML.Tests.UnitTests
{
  [TestClass]
  public class TestBase
  {
    public const double EPS = 0.0000001D;
    public const double EPS_ROUGH = 0.00001D;
    public const double GRAD_EPS = 0.000001D;
    public const double GRAD_STEP = 0.00000001D;

    private static TestContext m_Context;

    public static TestContext Context { get { return m_Context; } }

    [ClassInitialize]
    public static void BaseClassInit(TestContext context)
    {
      m_Context = context;
    }

    public static void AssertGradient(Func<double, double> f, double x, double actual)
    {
      var expected = (f(x+GRAD_STEP) - f(x-GRAD_STEP)) / (2*GRAD_STEP);
      Assert.IsTrue(Math.Abs(actual-expected)/expected < GRAD_EPS);
    }
  }
}
