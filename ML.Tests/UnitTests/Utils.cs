using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ML.Tests.UnitTests
{
  public static class MLAssert
  {
    public static void AreEpsEqual(double expected, double actual, double eps)
    {
      Assert.IsTrue(Math.Abs(expected-actual) < eps);
    }

    public static void AreEpsEqual(float expected, float actual, float eps)
    {
      Assert.IsTrue(Math.Abs(expected-actual) < eps);
    }
  }
}
