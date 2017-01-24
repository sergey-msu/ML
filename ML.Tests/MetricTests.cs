using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.Metric;
using ML.Core;

namespace ML.Tests
{
  [TestClass]
  public class MetricTests
  {
    [TestMethod]
    [ExpectedException(typeof(InvalidOperationException))]
    public void EuclideanMetric_Dist_Validation()
    {
      var metric = new EuclideanMetric();
      var p1 = new Point(1.0F, 0.0F);
      var p2 = new Point(0.0F, 1.0F, 3.0F);

      metric.Dist(p1, p2);
    }

    [TestMethod]
    public void EuclideanMetric_Dist_Test()
    {
      var eps = 0.0000001F;
      var metric = new EuclideanMetric();
      var p1 = new Point(1.0F, 0.0F);
      var p2 = new Point(0.0F, 1.0F);
      var p3 = new Point(1.0F, 3.0F);

      var d12 = metric.Dist(p1, p2);
      var d21 = metric.Dist(p2, p1);
      var d13 = metric.Dist(p1, p3);
      var d31 = metric.Dist(p3, p1);
      var d23 = metric.Dist(p2, p3);
      var d32 = metric.Dist(p3, p2);

      Assert.IsTrue(Math.Abs(d12 - Math.Sqrt(2)) < eps);
      Assert.IsTrue(Math.Abs(d21 - Math.Sqrt(2)) < eps);
      Assert.IsTrue(Math.Abs(d13 - 3.0F) < eps);
      Assert.IsTrue(Math.Abs(d31 - 3.0F) < eps);
      Assert.IsTrue(Math.Abs(d32 - Math.Sqrt(5)) < eps);
      Assert.IsTrue(Math.Abs(d23 - Math.Sqrt(5)) < eps);
    }
  }
}
