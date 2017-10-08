using System;

namespace Math.ODE
{
  public class Euler
  {
    #region Static

    public static class ODESolver
    {
      public static void Step1D(double x, double y,
                                Func<double, double, double> f,
                                double h,
                                out double r)
      {
        r = y + h*f(x, y);
      }

      public static void Step2D(double x, double y1, double y2,
                                Func<double, double, double, double> f1, Func<double, double, double, double> f2,
                                double h,
                                out double r1, out double r2)
      {
        r1 = y1 + h*f1(x, y1, y2);
        r2 = y2 + h*f2(x, y1, y2);
      }

      public static void Step3D(double x, double y1, double y2, double y3,
                                Func<double, double, double, double, double> f1, Func<double, double, double, double, double> f2, Func<double, double, double, double, double> f3,
                                double h,
                                out double r1, out double r2, out double r3)
      {
        r1 = y1 + h*f1(x, y1, y2, y3);
        r2 = y2 + h*f2(x, y1, y2, y3);
        r3 = y3 + h*f3(x, y1, y2, y3);
      }

      public static void Step(double x, double[] y,
                              Func<double, double[], double[]> f,
                              double h,
                              double[] r)
      {
        var n = y.Length;
        var val = f(x, y);
        for (int i=0; i<n; i++)
          r[i] = y[i] + h*val[i];
      }
    }

    #endregion

  }
}
