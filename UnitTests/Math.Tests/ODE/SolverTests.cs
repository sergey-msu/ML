using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Math.ODE;

namespace Math.Tests.ODE
{
  [TestClass]
  public class SolverTests : TestBase
  {
    #region Euler

    [TestMethod]
    public void EulerODESolver_Step1D()
    {
      // Arrange
      var h  = 0.1D;
      var x0 = 0;
      var y0 = 1;
      var f  = new Func<double, double, double>((x, y) => x + y*y);

      // Act
      double r;
      Euler.ODESolver.Step1D(x0, y0, f, h, out r);

      // Assert
      Assert.AreEqual(1.1D, r, EPS_DOUBLE);
    }

    [TestMethod]
    public void EulerODESolver_Step2D()
    {
      // Arrange
      var h   = 0.1D;
      var x0  = 1;
      var y10 = 1;
      var y20 = 2;
      var f1  = new Func<double, double, double, double>((x, y1, y2) => x + y1*y2);
      var f2  = new Func<double, double, double, double>((x, y1, y2) => x*y2);

      // Act
      double r1;
      double r2;
      Euler.ODESolver.Step2D(x0, y10, y20, f1, f2, h, out r1, out r2);

      // Assert
      Assert.AreEqual(1.3D, r1, EPS_DOUBLE);
      Assert.AreEqual(2.2D, r2, EPS_DOUBLE);
    }

    [TestMethod]
    public void EulerODESolver_Step3D()
    {
      // Arrange
      var h   = 0.1D;
      var x0  = 1;
      var y10 = 1;
      var y20 = 2;
      var y30 = 3;
      var f1  = new Func<double, double, double, double, double>((x, y1, y2, y3) => x + y1*y2*y3);
      var f2  = new Func<double, double, double, double, double>((x, y1, y2, y3) => x*y2 + y3);
      var f3  = new Func<double, double, double, double, double>((x, y1, y2, y3) => x - y2);

      // Act
      double r1;
      double r2;
      double r3;
      Euler.ODESolver.Step3D(x0, y10, y20, y30, f1, f2, f3, h, out r1, out r2, out r3);

      // Assert
      Assert.AreEqual(1.7D, r1, EPS_DOUBLE);
      Assert.AreEqual(2.5D, r2, EPS_DOUBLE);
      Assert.AreEqual(2.9D, r3, EPS_DOUBLE);
    }

    [TestMethod]
    public void EulerODESolver_Step()
    {
      // Arrange
      var h  = 0.1D;
      var x0 = 1;
      var y0 = new double[] { 1, 2, 3 };
      var f  = new Func<double, double[], double[]>
               (
                 (x, y) =>
                 new[]
                 {
                   x + y[0]*y[1]*y[2],
                   x*y[1] + y[2],
                   x - y[1]
                 });

      // Act
      var r = new double[3];
      Euler.ODESolver.Step(x0, y0, f, h, r);

      // Assert
      Assert.AreEqual(1.7D, r[0], EPS_DOUBLE);
      Assert.AreEqual(2.5D, r[1], EPS_DOUBLE);
      Assert.AreEqual(2.9D, r[2], EPS_DOUBLE);
    }

    [TestMethod]
    public void EulerODESolver_Step_Exp()
    {
      // Arrange
      var x0 = 0.0D;
      var x1 = 1.0D;
      var N  = 100000000;
      var h  = (x1 - x0)/N;
      var y0 = new double[] { 1 };
      var f  = new Func<double, double[], double[]>((x, y) => new[] { y[0] });

      // Act
      var r = new double[1];
      for (int i=0; i<N; i++)
      {
        Euler.ODESolver.Step(x0, y0, f, h, r);
        x0 += h;
        y0[0] = r[0];
      }

      // Assert
      Assert.AreEqual(System.Math.Exp(1), r[0], EPS);
    }

    #endregion

    #region RungeKutta

    [TestMethod]
    public void RungeKuttaODESolver_Step1D()
    {
      // Arrange
      var h  = 0.1D;
      var x0 = 0;
      var y0 = 1;
      var f  = new Func<double, double, double>((x, y) => x + y*y);

      // Act
      double r;
      RungeKutta.ODESolver.Step1D(x0, y0, f, h, out r);

      // Assert
      Assert.AreEqual(1.1164918497125D, r, EPS_DOUBLE);
    }

    [TestMethod]
    public void RungeKuttaODESolver_Step2D()
    {
      // Arrange
      var h   = 0.1D;
      var x0  = 1;
      var y10 = 1;
      var y20 = 2;
      var f1  = new Func<double, double, double, double>((x, y1, y2) => x + y2);
      var f2  = new Func<double, double, double, double>((x, y1, y2) => y1);

      // Act
      double r1;
      double r2;
      RungeKutta.ODESolver.Step2D(x0, y10, y20, f1, f2, h, out r1, out r2);

      // Assert
      Assert.AreEqual(1.31050833333D, r1, EPS_DOUBLE);
      Assert.AreEqual(2.11534583333D, r2, EPS_DOUBLE);
    }

    [TestMethod]
    public void RungeKuttaODESolver_Step3D()
    {
      // Arrange
      var h   = 0.1D;
      var x0  = 1;
      var y10 = 1;
      var y20 = 2;
      var y30 = 3;
      var f1  = new Func<double, double, double, double, double>((x, y1, y2, y3) => x + y3);
      var f2  = new Func<double, double, double, double, double>((x, y1, y2, y3) => x*y1);
      var f3  = new Func<double, double, double, double, double>((x, y1, y2, y3) => y2);

      // Act
      double r1;
      double r2;
      double r3;
      RungeKutta.ODESolver.Step3D(x0, y10, y20, y30, f1, f2, f3, h, out r1, out r2, out r3);

      // Assert
      Assert.AreEqual(1.41518833333D, r1, EPS_DOUBLE);
      Assert.AreEqual(2.12687541667D, r2, EPS_DOUBLE);
      Assert.AreEqual(3.20587979167D, r3, EPS_DOUBLE);
    }

    [TestMethod]
    public void RungeKuttaODESolver_Step()
    {
      // Arrange
      var h  = 0.1D;
      var x0 = 1;
      var y0 = new double[] { 1, 2, 3 };
      var f  = new Func<double, double[], double[]>(
                 (x, y) => new []
                           {
                             x + y[2],
                             x*y[0],
                             y[1]
                           });

      // Act
      var r = new double[3];
      RungeKutta.ODESolver.Step(x0, y0, f, h, r);

      // Assert
      Assert.AreEqual(1.41518833333D, r[0], EPS_DOUBLE);
      Assert.AreEqual(2.12687541667D, r[1], EPS_DOUBLE);
      Assert.AreEqual(3.20587979167D, r[2], EPS_DOUBLE);
    }

    [TestMethod]
    public void RungeKuttaODESolver_Step_Exp()
    {
      // Arrange
      var x0 = 0.0D;
      var x1 = 1.0D;
      var N  = 25;
      var h  = (x1 - x0)/N;
      var y0 = new double[] { 1 };
      var buf = new double[1];
      var f  = new Func<double, double[], double[]>(
                 (x, y) => { buf[0] = y[0]; return buf; }
               );

      // Act
      var r = new double[1];
      for (int i=0; i<N; i++)
      {
        RungeKutta.ODESolver.Step(x0, y0, f, h, r);
        x0 += h;
        y0[0] = r[0];
      }

      // Assert
      Assert.AreEqual(System.Math.Exp(1), r[0], EPS);
    }

    #endregion
  }
}
