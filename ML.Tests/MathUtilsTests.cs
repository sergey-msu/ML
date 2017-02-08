using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.Mathematics;

namespace ML.Tests
{
  [TestClass]
  public class MathUtilsTests
  {
    public const double EPS = 0.0000001F;

    [TestMethod]
    public void EntropyH_Values()
    {
      Assert.IsTrue(Math.Abs(MathUtils.EntropyH(0) - 0.0F) < EPS);
      Assert.IsTrue(Math.Abs(MathUtils.EntropyH(0.1F) - 0.33219280948874F) < EPS);
      Assert.IsTrue(Math.Abs(MathUtils.EntropyH(0.2F) - 0.46438561897747F) < EPS);
      Assert.IsTrue(Math.Abs(MathUtils.EntropyH(0.3F) - 0.52108967824986F) < EPS);
      Assert.IsTrue(Math.Abs(MathUtils.EntropyH(0.4F) - 0.52877123795494F) < EPS);
      Assert.IsTrue(Math.Abs(MathUtils.EntropyH(0.5F) - 0.5F) < EPS);
      Assert.IsTrue(Math.Abs(MathUtils.EntropyH(0.6F) - 0.44217935649972F) < EPS);
      Assert.IsTrue(Math.Abs(MathUtils.EntropyH(0.7F) - 0.36020122098083F) < EPS);
      Assert.IsTrue(Math.Abs(MathUtils.EntropyH(0.8F) - 0.25754247590989F) < EPS);
      Assert.IsTrue(Math.Abs(MathUtils.EntropyH(0.9F) - 0.13680278410054F) < EPS);
      Assert.IsTrue(Math.Abs(MathUtils.EntropyH(1.0F) - 0.0F) < EPS);
    }
  }
}
