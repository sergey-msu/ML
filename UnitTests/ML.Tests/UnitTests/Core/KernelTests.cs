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

      Assert.AreEqual(0.01752830049F, kernel.Value(-2.5F), EPS);
      Assert.AreEqual(0.05399096651F, kernel.Value(-2.0F), EPS);
      Assert.AreEqual(0.12951759566F, kernel.Value(-1.5F), EPS);
      Assert.AreEqual(0.24197072451F, kernel.Value(-1.0F), EPS);
      Assert.AreEqual(0.35206532676F, kernel.Value(-0.5F), EPS);
      Assert.AreEqual( 0.3989422804F,  kernel.Value( 0.0F), EPS);
      Assert.AreEqual(0.35206532676F, kernel.Value( 0.5F), EPS);
      Assert.AreEqual(0.24197072451F, kernel.Value( 1.0F), EPS);
      Assert.AreEqual(0.12951759566F, kernel.Value( 1.5F), EPS);
      Assert.AreEqual(0.05399096651F, kernel.Value( 2.0F), EPS);
      Assert.AreEqual(0.01752830049F, kernel.Value( 2.5F), EPS);
    }

    [TestMethod]
    public void QuadraticKernel_Value()
    {
      var kernel = new QuadraticKernel();

      Assert.AreEqual(0,       kernel.Value(-1.5F),  EPS);
      Assert.AreEqual(0,       kernel.Value(-1.01F), EPS);
      Assert.AreEqual(0,       kernel.Value(-1.0F),  EPS);
      Assert.AreEqual(QuadraticKernel.COEFF*0.0199F, kernel.Value(-0.99F), EPS);
      Assert.AreEqual(QuadraticKernel.COEFF*0.75F,   kernel.Value(-0.5F),  EPS);
      Assert.AreEqual(QuadraticKernel.COEFF*0.96F,   kernel.Value(-0.2F),  EPS);
      Assert.AreEqual(QuadraticKernel.COEFF*1.0F,    kernel.Value( 0.0F),  EPS);
      Assert.AreEqual(QuadraticKernel.COEFF*0.96F,   kernel.Value( 0.2F),  EPS);
      Assert.AreEqual(QuadraticKernel.COEFF*0.75F,   kernel.Value( 0.5F),  EPS);
      Assert.AreEqual(QuadraticKernel.COEFF*0.0199F, kernel.Value( 0.99F), EPS);
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
      Assert.AreEqual(0.5F, kernel.Value(-0.99F), EPS);
      Assert.AreEqual(0.5F, kernel.Value(-0.5F),  EPS);
      Assert.AreEqual(0.5F, kernel.Value(-0.2F),  EPS);
      Assert.AreEqual(0.5F, kernel.Value( 0.0F),  EPS);
      Assert.AreEqual(0.5F, kernel.Value( 0.2F),  EPS);
      Assert.AreEqual(0.5F, kernel.Value( 0.5F),  EPS);
      Assert.AreEqual(0.5F, kernel.Value( 0.99F), EPS);
      Assert.AreEqual(0,    kernel.Value( 1.0F),  EPS);
      Assert.AreEqual(0,    kernel.Value( 1.01F), EPS);
      Assert.AreEqual(0,    kernel.Value( 1.5F),  EPS);
    }
  }
}
