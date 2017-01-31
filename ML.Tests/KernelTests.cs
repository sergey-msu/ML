using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.Kernels;

namespace ML.Tests
{
  [TestClass]
  public class KernelTests
  {
    public const double EPS = 0.0000001F;

    [TestMethod]
    public void GaussianKernel_Calculate()
    {
      var kernel = new GaussianKernel();

      Assert.IsTrue(Math.Abs(kernel.Calculate(-2.5F) - 0.00193045413F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-2.0F) - 0.01831563888F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.5F) - 0.10539922456F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.0F) - 0.36787944117F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.5F) - 0.77880078307F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.0F) - 1.0F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.5F) - 0.77880078307F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.0F) - 0.36787944117F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.5F) - 0.10539922456F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 2.0F) - 0.01831563888F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 2.5F) - 0.00193045413F) < EPS);
    }

    [TestMethod]
    public void QuadraticKernel_Calculate()
    {
      var kernel = new QuadraticKernel();

      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.5F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.01F)) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.0F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.99F) - 0.0199F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.5F)  - 0.75F)   < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.2F)  - 0.96F)   < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.0F)  - 1.0F)    < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.2F)  - 0.96F)   < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.5F)  - 0.75F)   < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.99F) - 0.0199F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.0F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.01F)) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.5F))  < EPS);
    }

    [TestMethod]
    public void QuarticKernel_Calculate()
    {
      var kernel = new QuarticKernel();

      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.5F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.01F)) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.0F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.99F) - 0.00039601F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.5F)  - 0.5625F)   < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.2F)  - 0.9216F)   < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.0F)  - 1.0F)    < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.2F)  - 0.9216F)   < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.5F)  - 0.5625F)   < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.99F) - 0.00039601F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.0F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.01F)) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.5F))  < EPS);
    }

    [TestMethod]
    public void TriangularKernel_Calculate()
    {
      var kernel = new TriangularKernel();

      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.5F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.01F)) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.0F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.99F) - 0.01F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.5F)  - 0.5F)  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.2F)  - 0.8F)  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.0F)  - 1.0F)  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.2F)  - 0.8F)  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.5F)  - 0.5F)  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.99F) - 0.01F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.0F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.01F)) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.5F))  < EPS);
    }

    [TestMethod]
    public void RectangularKernel_Calculate()
    {
      var kernel = new RectangularKernel();

      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.5F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.01F)) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-1.0F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.99F) - 1.0F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.5F)  - 1.0F)  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate(-0.2F)  - 1.0F)  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.0F)  - 1.0F)  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.2F)  - 1.0F)  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.5F)  - 1.0F)  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 0.99F) - 1.0F) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.0F))  < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.01F)) < EPS);
      Assert.IsTrue(Math.Abs(kernel.Calculate( 1.5F))  < EPS);
    }
  }
}
