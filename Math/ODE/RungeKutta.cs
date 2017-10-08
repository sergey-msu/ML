using System;
using System.Runtime.CompilerServices;

namespace Math.ODE
{
  public class RungeKutta
  {
    #region Static

    [ThreadStatic]
    private static double[] m_Buffer = new double[10000];


    public static class ODESolver
    {
      public static void Step1D(double x, double y,
                                Func<double, double, double> f,
                                double h,
                                out double r)
      {
        var sh = h/2;

        var k = f(x, y);
        r = k;

        k = f(x + sh, y + sh*k);
        r += 2*k;

        k = f(x + sh, y + sh*k);
        r += 2*k;

        k = f(x + h, y + h*k);
        r += k;

        r = y + r*h/6.0D;
      }

      public static void Step2D(double x, double y1, double y2,
                                Func<double, double, double, double> f1, Func<double, double, double, double> f2,
                                double h,
                                out double r1, out double r2)
      {
        var h2 = h/2;
        var h6 = h/6.0D;

        var k1 = f1(x, y1, y2);
        var k2 = f2(x, y1, y2);
        r1 = k1;
        r2 = k2;
        var buf1 = y1 + h2*k1;
        var buf2 = y2 + h2*k2;

        k1 = f1(x + h2, buf1, buf2);
        k2 = f2(x + h2, buf1, buf2);
        r1 += 2*k1;
        r2 += 2*k2;
        buf1 = y1 + h2*k1;
        buf2 = y2 + h2*k2;

        k1 = f1(x + h2, buf1, buf2);
        k2 = f2(x + h2, buf1, buf2);
        r1 += 2*k1;
        r2 += 2*k2;
        buf1 = y1 + h*k1;
        buf2 = y2 + h*k2;

        k1 = f1(x + h, buf1, buf2);
        k2 = f2(x + h, buf1, buf2);
        r1 += k1;
        r2 += k2;

        r1 = y1 + r1*h6;
        r2 = y2 + r2*h6;
      }

      public static void Step3D(double x, double y1, double y2, double y3,
                                Func<double, double, double, double, double> f1, Func<double, double, double, double, double> f2, Func<double, double, double, double, double> f3,
                                double h,
                                out double r1, out double r2, out double r3)
      {
        var h2 = h/2;
        var h6 = h/6.0D;

        var k1 = f1(x, y1, y2, y3);
        var k2 = f2(x, y1, y2, y3);
        var k3 = f3(x, y1, y2, y3);
        r1 = k1;
        r2 = k2;
        r3 = k3;
        var buf1 = y1 + h2*k1;
        var buf2 = y2 + h2*k2;
        var buf3 = y3 + h2*k3;

        k1 = f1(x + h2, buf1, buf2, buf3);
        k2 = f2(x + h2, buf1, buf2, buf3);
        k3 = f3(x + h2, buf1, buf2, buf3);
        r1 += 2*k1;
        r2 += 2*k2;
        r3 += 2*k3;
        buf1 = y1 + h2*k1;
        buf2 = y2 + h2*k2;
        buf3 = y3 + h2*k3;

        k1 = f1(x + h2, buf1, buf2, buf3);
        k2 = f2(x + h2, buf1, buf2, buf3);
        k3 = f3(x + h2, buf1, buf2, buf3);
        r1 += 2*k1;
        r2 += 2*k2;
        r3 += 2*k3;
        buf1 = y1 + h*k1;
        buf2 = y2 + h*k2;
        buf3 = y3 + h*k3;

        k1 = f1(x + h, buf1, buf2, buf3);
        k2 = f2(x + h, buf1, buf2, buf3);
        k3 = f3(x + h, buf1, buf2, buf3);
        r1 += k1;
        r2 += k2;
        r3 += k3;

        r1 = y1 + r1*h6;
        r2 = y2 + r2*h6;
        r3 = y3 + r3*h6;
      }

      public static void Step(double x, double[] y,
                              Func<double, double[], double[]> f,
                              double h,
                              double[] r)
      {
        var n = y.Length;
        var h2 = h/2;
        var h6 = h/6.0D;
        ensureBuffer(n);

        // k1
        var k = f(x, y);
        for (int i=0; i<n; i++)
        {
          r[i] = k[i];
          m_Buffer[i] = y[i] + h2*k[i];
        }

        // k2
        k = f(x + h2, m_Buffer);
        for (int i=0; i<n; i++)
        {
          r[i] += 2*k[i];
          m_Buffer[i] = y[i] + h2*k[i];
        }

        // k3
        k = f(x + h2, m_Buffer);
        for (int i=0; i<n; i++)
        {
          r[i] += 2*k[i];
          m_Buffer[i] = y[i] + h*k[i];
        }

        // k4
        k = f(x + h, m_Buffer);
        for (int i=0; i<n; i++)
        {
          r[i] += k[i];
          r[i] = y[i] + r[i]*h6;
        }
      }
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ensureBuffer(int n)
    {
      if (n > m_Buffer.Length)
        m_Buffer = new double[n];
    }

    #endregion

  }
}
