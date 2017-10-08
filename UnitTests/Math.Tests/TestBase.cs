using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Math.Tests
{
  [TestClass]
  public class TestBase
  {
    private Random m_Random = new Random(0);

    public const double EPS = 0.0000001D;
    public const double EPS_DOUBLE = 0.00000000001D;
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
  }
}
