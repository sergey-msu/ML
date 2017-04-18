using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.Kernels;

namespace ML.Tests.UnitTests.Core
{
  [TestClass]
  public class KernelTests : TestBase
  {
    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    [TestMethod]
    public void GaussianKernel_Value()
    {
      var kernel = new GaussianKernel();

      Assert.AreEqual(0.00193045413F, kernel.Value(-2.5F), EPS);
      Assert.AreEqual(0.01831563888F, kernel.Value(-2.0F), EPS);
      Assert.AreEqual(0.10539922456F, kernel.Value(-1.5F), EPS);
      Assert.AreEqual(0.36787944117F, kernel.Value(-1.0F), EPS);
      Assert.AreEqual(0.77880078307F, kernel.Value(-0.5F), EPS);
      Assert.AreEqual(1.0F,           kernel.Value( 0.0F), EPS);
      Assert.AreEqual(0.77880078307F, kernel.Value( 0.5F), EPS);
      Assert.AreEqual(0.36787944117F, kernel.Value( 1.0F), EPS);
      Assert.AreEqual(0.10539922456F, kernel.Value( 1.5F), EPS);
      Assert.AreEqual(0.01831563888F, kernel.Value( 2.0F), EPS);
      Assert.AreEqual(0.00193045413F, kernel.Value( 2.5F), EPS);
    }

    [TestMethod]
    public void QuadraticKernel_Value()
    {
      var kernel = new QuadraticKernel();

      Assert.AreEqual(0,       kernel.Value(-1.5F),  EPS);
      Assert.AreEqual(0,       kernel.Value(-1.01F), EPS);
      Assert.AreEqual(0,       kernel.Value(-1.0F),  EPS);
      Assert.AreEqual(0.0199F, kernel.Value(-0.99F), EPS);
      Assert.AreEqual(0.75F,   kernel.Value(-0.5F),  EPS);
      Assert.AreEqual(0.96F,   kernel.Value(-0.2F),  EPS);
      Assert.AreEqual(1.0F,    kernel.Value( 0.0F),  EPS);
      Assert.AreEqual(0.96F,   kernel.Value( 0.2F),  EPS);
      Assert.AreEqual(0.75F,   kernel.Value( 0.5F),  EPS);
      Assert.AreEqual(0.0199F, kernel.Value( 0.99F), EPS);
      Assert.AreEqual(0,       kernel.Value( 1.0F),  EPS);
      Assert.AreEqual(0,       kernel.Value( 1.01F), EPS);
      Assert.AreEqual(0,       kernel.Value( 1.5F),  EPS);
    }

    [TestMethod]
    public void QuarticKernel_Value()
    {
      var kernel = new QuarticKernel();

      Assert.AreEqual(0,           kernel.Value(-1.5F),  EPS);
      Assert.AreEqual(0,           kernel.Value(-1.01F), EPS);
      Assert.AreEqual(0,           kernel.Value(-1.0F),  EPS);
      Assert.AreEqual(0.00039601F, kernel.Value(-0.99F), EPS);
      Assert.AreEqual(0.5625F,     kernel.Value(-0.5F),  EPS);
      Assert.AreEqual(0.9216F,     kernel.Value(-0.2F),  EPS);
      Assert.AreEqual(1.0F,        kernel.Value( 0.0F),  EPS);
      Assert.AreEqual(0.9216F,     kernel.Value( 0.2F),  EPS);
      Assert.AreEqual(0.5625F,     kernel.Value( 0.5F),  EPS);
      Assert.AreEqual(0.00039601F, kernel.Value( 0.99F), EPS);
      Assert.AreEqual(0,           kernel.Value( 1.0F),  EPS);
      Assert.AreEqual(0,           kernel.Value( 1.01F), EPS);
      Assert.AreEqual(0,           kernel.Value( 1.5F),  EPS);
    }

    [TestMethod]
    public void TriangularKernel_Value()
    {
      var kernel = new TriangularKernel();

      Assert.AreEqual(0,     kernel.Value(-1.5F),  EPS);
      Assert.AreEqual(0,     kernel.Value(-1.01F), EPS);
      Assert.AreEqual(0,     kernel.Value(-1.0F),  EPS);
      Assert.AreEqual(0.01F, kernel.Value(-0.99F), EPS);
      Assert.AreEqual(0.5F,  kernel.Value(-0.5F),  EPS);
      Assert.AreEqual(0.8F,  kernel.Value(-0.2F),  EPS);
      Assert.AreEqual(1.0F,  kernel.Value( 0.0F),  EPS);
      Assert.AreEqual(0.8F,  kernel.Value( 0.2F),  EPS);
      Assert.AreEqual(0.5F,  kernel.Value( 0.5F),  EPS);
      Assert.AreEqual(0.01F, kernel.Value( 0.99F), EPS);
      Assert.AreEqual(0,     kernel.Value( 1.0F),  EPS);
      Assert.AreEqual(0,     kernel.Value( 1.01F), EPS);
      Assert.AreEqual(0,     kernel.Value( 1.5F),  EPS);
    }

    [TestMethod]
    public void RectangularKernel_Value()
    {
      var kernel = new RectangularKernel();

      Assert.AreEqual(0,    kernel.Value(-1.5F),  EPS);
      Assert.AreEqual(0,    kernel.Value(-1.01F), EPS);
      Assert.AreEqual(0,    kernel.Value(-1.0F),  EPS);
      Assert.AreEqual(1.0F, kernel.Value(-0.99F), EPS);
      Assert.AreEqual(1.0F, kernel.Value(-0.5F),  EPS);
      Assert.AreEqual(1.0F, kernel.Value(-0.2F),  EPS);
      Assert.AreEqual(1.0F, kernel.Value( 0.0F),  EPS);
      Assert.AreEqual(1.0F, kernel.Value( 0.2F),  EPS);
      Assert.AreEqual(1.0F, kernel.Value( 0.5F),  EPS);
      Assert.AreEqual(1.0F, kernel.Value( 0.99F), EPS);
      Assert.AreEqual(0,    kernel.Value( 1.0F),  EPS);
      Assert.AreEqual(0,    kernel.Value( 1.01F), EPS);
      Assert.AreEqual(0,    kernel.Value( 1.5F),  EPS);
    }
  }
}
