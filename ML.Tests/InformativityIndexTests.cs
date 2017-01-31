using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.Logical;
using ML.Core;

namespace ML.Tests
{
  [TestClass]
  public class InformativityIndexTests
  {
    #region Setup

      public static readonly Class[] CLASSES = new Class[]
                                               {
                                                 new Class("A", 1),
                                                 new Class("B", 2),
                                                 new Class("C", 3)
                                               };

      public static readonly ClassifiedSample SAMPLE_2C = new ClassifiedSample
      {
        { new Point(0.0F), CLASSES[0] },
        { new Point(1.0F), CLASSES[0] },
        { new Point(2.0F), CLASSES[1] },
        { new Point(3.0F), CLASSES[1] },
        { new Point(4.0F), CLASSES[1] },
      };

      public static readonly ClassifiedSample SAMPLE_3C = new ClassifiedSample
      {
        { new Point(0.0F), CLASSES[0] },
        { new Point(1.0F), CLASSES[0] },
        { new Point(2.0F), CLASSES[1] },
        { new Point(3.0F), CLASSES[1] },
        { new Point(4.0F), CLASSES[2] },
        { new Point(5.0F), CLASSES[2] }
      };

      public static Predicate<Point> SimplePattern(int idx, double a) { return p => p[idx] < a; }

    #endregion

    public const double EPS = 0.0000001F;

    [TestMethod]
    public void GiniIndex_Calculate_2C()
    {
      var index = new GiniIndex();

      var p1 = SimplePattern(0, -0.5F);
      var p2 = SimplePattern(0,  0.5F);
      var p3 = SimplePattern(0,  1.5F);
      var p4 = SimplePattern(0,  2.5F);
      var p5 = SimplePattern(0,  3.5F);
      var p6 = SimplePattern(0,  4.5F);

      var i1 = index.Calculate(p1, SAMPLE_2C);
      var i2 = index.Calculate(p2, SAMPLE_2C);
      var i3 = index.Calculate(p3, SAMPLE_2C);
      var i4 = index.Calculate(p4, SAMPLE_2C);
      var i5 = index.Calculate(p5, SAMPLE_2C);
      var i6 = index.Calculate(p6, SAMPLE_2C);

      Assert.AreEqual(i1, 4);
      Assert.AreEqual(i2, 3);
      Assert.AreEqual(i3, 4);
      Assert.AreEqual(i4, 2);
      Assert.AreEqual(i5, 2);
      Assert.AreEqual(i6, 4);
    }

    [TestMethod]
    public void GiniIndex_Calculate_3C()
    {
      var index = new GiniIndex();

      var p1  = SimplePattern(0, -0.5F);
      var p2  = SimplePattern(0,  0.5F);
      var p3  = SimplePattern(0,  1.5F);
      var p4  = SimplePattern(0,  2.5F);
      var p5  = SimplePattern(0,  3.5F);
      var p6  = SimplePattern(0,  4.5F);
      var p7  = SimplePattern(0,  5.5F);

      var i1  = index.Calculate(p1, SAMPLE_3C);
      var i2  = index.Calculate(p2, SAMPLE_3C);
      var i3  = index.Calculate(p3, SAMPLE_3C);
      var i4  = index.Calculate(p4, SAMPLE_3C);
      var i5  = index.Calculate(p5, SAMPLE_3C);
      var i6  = index.Calculate(p6, SAMPLE_3C);
      var i7  = index.Calculate(p7, SAMPLE_3C);

      Assert.AreEqual(i1, 3);
      Assert.AreEqual(i2, 2);
      Assert.AreEqual(i3, 3);
      Assert.AreEqual(i4, 2);
      Assert.AreEqual(i5, 3);
      Assert.AreEqual(i6, 2);
      Assert.AreEqual(i7, 3);
    }

    [TestMethod]
    public void DonskoyIndex_Calculate_2C()
    {
      var index = new DonskoyIndex();

      var p1 = SimplePattern(0, -0.5F);
      var p2 = SimplePattern(0,  0.5F);
      var p3 = SimplePattern(0,  1.5F);
      var p4 = SimplePattern(0,  2.5F);
      var p5 = SimplePattern(0,  3.5F);
      var p6 = SimplePattern(0,  4.5F);

      var i1 = index.Calculate(p1, SAMPLE_2C);
      var i2 = index.Calculate(p2, SAMPLE_2C);
      var i3 = index.Calculate(p3, SAMPLE_2C);
      var i4 = index.Calculate(p4, SAMPLE_2C);
      var i5 = index.Calculate(p5, SAMPLE_2C);
      var i6 = index.Calculate(p6, SAMPLE_2C);

      Assert.AreEqual(i1, 0);
      Assert.AreEqual(i2, 3);
      Assert.AreEqual(i3, 6);
      Assert.AreEqual(i4, 4);
      Assert.AreEqual(i5, 2);
      Assert.AreEqual(i6, 0);
    }

    [TestMethod]
    public void DonskoyIndex_Calculate_3C()
    {
      var index = new DonskoyIndex();

      var p1  = SimplePattern(0, -0.5F);
      var p2  = SimplePattern(0,  0.5F);
      var p3  = SimplePattern(0,  1.5F);
      var p4  = SimplePattern(0,  2.5F);
      var p5  = SimplePattern(0,  3.5F);
      var p6  = SimplePattern(0,  4.5F);
      var p7  = SimplePattern(0,  5.5F);

      var i1  = index.Calculate(p1, SAMPLE_3C);
      var i2  = index.Calculate(p2, SAMPLE_3C);
      var i3  = index.Calculate(p3, SAMPLE_3C);
      var i4  = index.Calculate(p4, SAMPLE_3C);
      var i5  = index.Calculate(p5, SAMPLE_3C);
      var i6  = index.Calculate(p6, SAMPLE_3C);
      var i7  = index.Calculate(p7, SAMPLE_3C);

      Assert.AreEqual(i1, 0);
      Assert.AreEqual(i2, 4);
      Assert.AreEqual(i3, 8);
      Assert.AreEqual(i4, 8);
      Assert.AreEqual(i5, 8);
      Assert.AreEqual(i6, 4);
      Assert.AreEqual(i7, 0);
    }

    [TestMethod]
    public void EntropyIndex_Calculate_2C()
    {
      var index = new EntropyIndex();

      var p1 = SimplePattern(0, -0.5F);
      var p2 = SimplePattern(0,  0.5F);
      var p3 = SimplePattern(0,  1.5F);
      var p4 = SimplePattern(0,  2.5F);
      var p5 = SimplePattern(0,  3.5F);
      var p6 = SimplePattern(0,  4.5F);

      var i1 = index.Calculate(p1, SAMPLE_2C);
      var i2 = index.Calculate(p2, SAMPLE_2C);
      var i3 = index.Calculate(p3, SAMPLE_2C);
      var i4 = index.Calculate(p4, SAMPLE_2C);
      var i5 = index.Calculate(p5, SAMPLE_2C);
      var i6 = index.Calculate(p6, SAMPLE_2C);

      Assert.IsTrue(Math.Abs(i1) < EPS);
      Assert.IsTrue(Math.Abs(i2-0.3219280F) < EPS);
      Assert.IsTrue(Math.Abs(i3-0.9709505F) < EPS);
      Assert.IsTrue(Math.Abs(i4-0.419973075F) < EPS);
      Assert.IsTrue(Math.Abs(i5-0.170950577F) < EPS);
      Assert.IsTrue(Math.Abs(i6) < EPS);
    }

    [TestMethod]
    public void EntropyIndex_Calculate_3C()
    {
      var index = new EntropyIndex();

      var p1  = SimplePattern(0, -0.5F);
      var p2  = SimplePattern(0,  0.5F);
      var p3  = SimplePattern(0,  1.5F);
      var p4  = SimplePattern(0,  2.5F);
      var p5  = SimplePattern(0,  3.5F);
      var p6  = SimplePattern(0,  4.5F);
      var p7  = SimplePattern(0,  5.5F);

      var i1  = index.Calculate(p1, SAMPLE_3C);
      var i2  = index.Calculate(p2, SAMPLE_3C);
      var i3  = index.Calculate(p3, SAMPLE_3C);
      var i4  = index.Calculate(p4, SAMPLE_3C);
      var i5  = index.Calculate(p5, SAMPLE_3C);
      var i6  = index.Calculate(p6, SAMPLE_3C);
      var i7  = index.Calculate(p7, SAMPLE_3C);

      Assert.IsTrue(Math.Abs(i1) < EPS);
      Assert.IsTrue(Math.Abs(i2-0.31668908F) < EPS);
      Assert.IsTrue(Math.Abs(i3-0.9182958F) < EPS);
      Assert.IsTrue(Math.Abs(i4-0.6666667F) < EPS);
      Assert.IsTrue(Math.Abs(i5-0.9182958F) < EPS);
      Assert.IsTrue(Math.Abs(i6-0.316689074F) < EPS);
      Assert.IsTrue(Math.Abs(i7) < EPS);
    }
  }
}
