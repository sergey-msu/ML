using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.Metric;
using ML.Core;

namespace ML.Tests
{
  [TestClass]
  public class MetricTests
  {
    public const double EPS = 0.0000001F;

    [TestMethod]
    [ExpectedException(typeof(MLException))]
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

      Assert.IsTrue(Math.Abs(d12 - Math.Sqrt(2)) < EPS);
      Assert.IsTrue(Math.Abs(d21 - Math.Sqrt(2)) < EPS);
      Assert.IsTrue(Math.Abs(d13 - 3.0F) < EPS);
      Assert.IsTrue(Math.Abs(d31 - 3.0F) < EPS);
      Assert.IsTrue(Math.Abs(d32 - Math.Sqrt(5)) < EPS);
      Assert.IsTrue(Math.Abs(d23 - Math.Sqrt(5)) < EPS);
    }

    [TestMethod]
    public void EuclideanMetric_Dist2_Test()
    {
      var metric = new EuclideanMetric();
      var p1 = new Point(1.0F, 0.0F);
      var p2 = new Point(0.0F, 1.0F);
      var p3 = new Point(1.0F, 3.0F);

      var d12 = metric.Dist2(p1, p2);
      var d21 = metric.Dist2(p2, p1);
      var d13 = metric.Dist2(p1, p3);
      var d31 = metric.Dist2(p3, p1);
      var d23 = metric.Dist2(p2, p3);
      var d32 = metric.Dist2(p3, p2);

      Assert.IsTrue(Math.Abs(d12 - 2.0F) < EPS);
      Assert.IsTrue(Math.Abs(d21 - 2.0F) < EPS);
      Assert.IsTrue(Math.Abs(d13 - 9.0F) < EPS);
      Assert.IsTrue(Math.Abs(d31 - 9.0F) < EPS);
      Assert.IsTrue(Math.Abs(d32 - 5.0F) < EPS);
      Assert.IsTrue(Math.Abs(d23 - 5.0F) < EPS);
    }

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void LInftyMetric_Dist_Validation()
    {
      var metric = new LInftyMetric();
      var p1 = new Point(1.0F, 0.0F);
      var p2 = new Point(0.0F, 1.0F, 3.0F);

      metric.Dist(p1, p2);
    }

    [TestMethod]
    public void LInftyMetric_Dist_Test()
    {
      var metric = new LInftyMetric();
      var p1 = new Point(1.0F, 0.0F);
      var p2 = new Point(0.0F, 1.0F);
      var p3 = new Point(1.0F, 3.0F);

      var d12 = metric.Dist(p1, p2);
      var d21 = metric.Dist(p2, p1);
      var d13 = metric.Dist(p1, p3);
      var d31 = metric.Dist(p3, p1);
      var d23 = metric.Dist(p2, p3);
      var d32 = metric.Dist(p3, p2);

      Assert.IsTrue(Math.Abs(d12 - 1.0F) < EPS);
      Assert.IsTrue(Math.Abs(d21 - 1.0F) < EPS);
      Assert.IsTrue(Math.Abs(d13 - 3.0F) < EPS);
      Assert.IsTrue(Math.Abs(d31 - 3.0F) < EPS);
      Assert.IsTrue(Math.Abs(d32 - 2.0F) < EPS);
      Assert.IsTrue(Math.Abs(d23 - 2.0F) < EPS);
    }

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void LpMetric_Dist_Validation()
    {
      var metric = new LpMetric(1.2F);
      var p1 = new Point(1.0F, 0.0F);
      var p2 = new Point(0.0F, 1.0F, 3.0F);

      metric.Dist(p1, p2);
    }

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void LpMetric_Dist_Ctor()
    {
      var metric = new LpMetric(0.8F);
      var p1 = new Point(1.0F, 0.0F);
      var p2 = new Point(0.0F, 1.0F);

      metric.Dist(p1, p2);
    }

    [TestMethod]
    public void LpMetric_Dist_Test()
    {
      var p = 3.8F;
      var metric1 = new LpMetric(1.0F);
      var metric2 = new LpMetric(p);
      var p1 = new Point(1.0F, 0.0F);
      var p2 = new Point(0.0F, 1.0F);
      var p3 = new Point(1.0F, 3.0F);

      var d12 = metric1.Dist(p1, p2);
      var d21 = metric1.Dist(p2, p1);
      var d13 = metric1.Dist(p1, p3);
      var d31 = metric1.Dist(p3, p1);
      var d23 = metric1.Dist(p2, p3);
      var d32 = metric1.Dist(p3, p2);

      Assert.IsTrue(Math.Abs(d12 - 2.0F) < EPS);
      Assert.IsTrue(Math.Abs(d21 - 2.0F) < EPS);
      Assert.IsTrue(Math.Abs(d13 - 3.0F) < EPS);
      Assert.IsTrue(Math.Abs(d31 - 3.0F) < EPS);
      Assert.IsTrue(Math.Abs(d32 - 3.0F) < EPS);
      Assert.IsTrue(Math.Abs(d23 - 3.0F) < EPS);

      d12 = metric2.Dist(p1, p2);
      d21 = metric2.Dist(p2, p1);
      d13 = metric2.Dist(p1, p3);
      d31 = metric2.Dist(p3, p1);
      d23 = metric2.Dist(p2, p3);
      d32 = metric2.Dist(p3, p2);

      Assert.IsTrue(Math.Abs(d12 - Math.Pow(2.0D, 1.0D/p)) < EPS);
      Assert.IsTrue(Math.Abs(d21 - Math.Pow(2.0D, 1.0D/p)) < EPS);
      Assert.IsTrue(Math.Abs(d13 - 3.0F) < EPS);
      Assert.IsTrue(Math.Abs(d31 - 3.0F) < EPS);
      Assert.IsTrue(Math.Abs(d23 - Math.Pow(1 + Math.Pow(2.0D, p), 1.0D/p)) < EPS);
      Assert.IsTrue(Math.Abs(d32 - Math.Pow(1 + Math.Pow(2.0D, p), 1.0D/p)) < EPS);
    }
  }
}
