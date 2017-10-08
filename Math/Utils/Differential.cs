using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Math.Utils
{
  public static class Differential
  {
    public static void Derivative(double[] t, double[] x, int order, double[] result)
    {
      if (order==1) doFirstDerivative(t, x, result);

      // throw
    }

    public static void Curvature(double[] x, double[] y, double[] output)
    {
      var n = x.Length;

      for (int i=1; i<n-1; i++)
      {
        var x0 = x[i-1];
        var x1 = x[i];
        var x2 = x[i+1];
        var y0 = y[i-1];
        var y1 = y[i];
        var y2 = y[i+1];

        var a = 8*((y2 - y1)*(x1 - x0) - (x2 - x1)*(y1 - y0));
        var b = (x2 - x0)*(x2 - x0) + (y2 - y0)*(y2 - y0);
        output[i] = System.Math.Abs(a)/System.Math.Sqrt(b*b*b);
      }

      output[0] = 2*output[1] - output[2];
      output[n-1] = 2*output[n-2] - output[n-3];
    }

    public static void Angles(double[] x, double[] y, double[] output)
    {
      var n = x.Length;

      output[0] = 0;
      output[n-1] = 0;
      var dx1 = x[1] - x[0];
      var dy1 = y[1] - y[0];
      var dr1 = System.Math.Sqrt(dx1*dx1 + dy1*dy1);

      for (int i=1; i<n-1; i++)
      {
        var dx2 = x[i+1] - x[i];
        var dy2 = y[i+1] - y[i];
        var dr2 = System.Math.Sqrt(dx2*dx2 + dy2*dy2);

        var a = System.Math.Acos((dx1*dx2 + dy1*dy2)/(dr1*dr2));
        output[i] = a;

        dx1 = dx2;
        dy1 = dy2;
        dr1 = dr2;
      }
    }

    #region .pvt

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void doFirstDerivative(double[] t, double[] x, double[] result)
    {
      var n = x.Length;
      for (int i=1; i<n-1; i++)
      {
        var dt1 = t[i+1] - t[i];
        var dt2 = t[i] - t[i-1];
        result[i] = (x[i+1] - x[i-1])/(dt1 + dt2);
      }

      result[0] = (x[1] - x[0])/(t[1] - t[0]);
      result[n-1] = (x[n-1] - x[n-2])/(t[n-1] - t[n-2]);
    }

    #endregion
  }
}
