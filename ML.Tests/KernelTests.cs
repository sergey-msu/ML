using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.Kernels;

namespace ML.Tests
{
  [TestClass]
  public class KernelTests : TestBase
  {
    public const double EPS = 0.0000001D;

    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    [TestMethod]
    public void GaussianKernel_Value()
    {
      var kernel = new GaussianKernel();

      MLAssert.AreEpsEqual(0.00193045413F, kernel.Value(-2.5F), EPS);
      MLAssert.AreEpsEqual(0.01831563888F, kernel.Value(-2.0F), EPS);
      MLAssert.AreEpsEqual(0.10539922456F, kernel.Value(-1.5F), EPS);
      MLAssert.AreEpsEqual(0.36787944117F, kernel.Value(-1.0F), EPS);
      MLAssert.AreEpsEqual(0.77880078307F, kernel.Value(-0.5F), EPS);
      MLAssert.AreEpsEqual(1.0F,           kernel.Value( 0.0F), EPS);
      MLAssert.AreEpsEqual(0.77880078307F, kernel.Value( 0.5F), EPS);
      MLAssert.AreEpsEqual(0.36787944117F, kernel.Value( 1.0F), EPS);
      MLAssert.AreEpsEqual(0.10539922456F, kernel.Value( 1.5F), EPS);
      MLAssert.AreEpsEqual(0.01831563888F, kernel.Value( 2.0F), EPS);
      MLAssert.AreEpsEqual(0.00193045413F, kernel.Value( 2.5F), EPS);
    }

    [TestMethod]
    public void QuadraticKernel_Value()
    {
      var kernel = new QuadraticKernel();

      MLAssert.AreEpsEqual(0,       kernel.Value(-1.5F),  EPS);
      MLAssert.AreEpsEqual(0,       kernel.Value(-1.01F), EPS);
      MLAssert.AreEpsEqual(0,       kernel.Value(-1.0F),  EPS);
      MLAssert.AreEpsEqual(0.0199F, kernel.Value(-0.99F), EPS);
      MLAssert.AreEpsEqual(0.75F,   kernel.Value(-0.5F),  EPS);
      MLAssert.AreEpsEqual(0.96F,   kernel.Value(-0.2F),  EPS);
      MLAssert.AreEpsEqual(1.0F,    kernel.Value( 0.0F),  EPS);
      MLAssert.AreEpsEqual(0.96F,   kernel.Value( 0.2F),  EPS);
      MLAssert.AreEpsEqual(0.75F,   kernel.Value( 0.5F),  EPS);
      MLAssert.AreEpsEqual(0.0199F, kernel.Value( 0.99F), EPS);
      MLAssert.AreEpsEqual(0,       kernel.Value( 1.0F),  EPS);
      MLAssert.AreEpsEqual(0,       kernel.Value( 1.01F), EPS);
      MLAssert.AreEpsEqual(0,       kernel.Value( 1.5F),  EPS);
    }

    [TestMethod]
    public void QuarticKernel_Value()
    {
      var kernel = new QuarticKernel();

      MLAssert.AreEpsEqual(0,           kernel.Value(-1.5F),  EPS);
      MLAssert.AreEpsEqual(0,           kernel.Value(-1.01F), EPS);
      MLAssert.AreEpsEqual(0,           kernel.Value(-1.0F),  EPS);
      MLAssert.AreEpsEqual(0.00039601F, kernel.Value(-0.99F), EPS);
      MLAssert.AreEpsEqual(0.5625F,     kernel.Value(-0.5F),  EPS);
      MLAssert.AreEpsEqual(0.9216F,     kernel.Value(-0.2F),  EPS);
      MLAssert.AreEpsEqual(1.0F,        kernel.Value( 0.0F),  EPS);
      MLAssert.AreEpsEqual(0.9216F,     kernel.Value( 0.2F),  EPS);
      MLAssert.AreEpsEqual(0.5625F,     kernel.Value( 0.5F),  EPS);
      MLAssert.AreEpsEqual(0.00039601F, kernel.Value( 0.99F), EPS);
      MLAssert.AreEpsEqual(0,           kernel.Value( 1.0F),  EPS);
      MLAssert.AreEpsEqual(0,           kernel.Value( 1.01F), EPS);
      MLAssert.AreEpsEqual(0,           kernel.Value( 1.5F),  EPS);
    }

    [TestMethod]
    public void TriangularKernel_Value()
    {
      var kernel = new TriangularKernel();

      MLAssert.AreEpsEqual(0,     kernel.Value(-1.5F),  EPS);
      MLAssert.AreEpsEqual(0,     kernel.Value(-1.01F), EPS);
      MLAssert.AreEpsEqual(0,     kernel.Value(-1.0F),  EPS);
      MLAssert.AreEpsEqual(0.01F, kernel.Value(-0.99F), EPS);
      MLAssert.AreEpsEqual(0.5F,  kernel.Value(-0.5F),  EPS);
      MLAssert.AreEpsEqual(0.8F,  kernel.Value(-0.2F),  EPS);
      MLAssert.AreEpsEqual(1.0F,  kernel.Value( 0.0F),  EPS);
      MLAssert.AreEpsEqual(0.8F,  kernel.Value( 0.2F),  EPS);
      MLAssert.AreEpsEqual(0.5F,  kernel.Value( 0.5F),  EPS);
      MLAssert.AreEpsEqual(0.01F, kernel.Value( 0.99F), EPS);
      MLAssert.AreEpsEqual(0,     kernel.Value( 1.0F),  EPS);
      MLAssert.AreEpsEqual(0,     kernel.Value( 1.01F), EPS);
      MLAssert.AreEpsEqual(0,     kernel.Value( 1.5F),  EPS);
    }

    [TestMethod]
    public void RectangularKernel_Value()
    {
      var kernel = new RectangularKernel();

      MLAssert.AreEpsEqual(0,    kernel.Value(-1.5F),  EPS);
      MLAssert.AreEpsEqual(0,    kernel.Value(-1.01F), EPS);
      MLAssert.AreEpsEqual(0,    kernel.Value(-1.0F),  EPS);
      MLAssert.AreEpsEqual(1.0F, kernel.Value(-0.99F), EPS);
      MLAssert.AreEpsEqual(1.0F, kernel.Value(-0.5F),  EPS);
      MLAssert.AreEpsEqual(1.0F, kernel.Value(-0.2F),  EPS);
      MLAssert.AreEpsEqual(1.0F, kernel.Value( 0.0F),  EPS);
      MLAssert.AreEpsEqual(1.0F, kernel.Value( 0.2F),  EPS);
      MLAssert.AreEpsEqual(1.0F, kernel.Value( 0.5F),  EPS);
      MLAssert.AreEpsEqual(1.0F, kernel.Value( 0.99F), EPS);
      MLAssert.AreEpsEqual(0,    kernel.Value( 1.0F),  EPS);
      MLAssert.AreEpsEqual(0,    kernel.Value( 1.01F), EPS);
      MLAssert.AreEpsEqual(0,    kernel.Value( 1.5F),  EPS);
    }
  }
}
