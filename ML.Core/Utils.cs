using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace ML.Core
{
  public static class Utils
  {
    #region Inner

    public struct Point2D
    {
      public Point2D(double x, double y)
      {
        X = x;
        Y = y;
      }

      public readonly double X;
      public readonly double Y;

      public Point2D? ToBoxMuller()
      {
        var s = X * X + Y * Y;
        if (s > 1 || s == 0) return null;

        var L = Math.Sqrt(-2 * Math.Log(s) / s);

        return new Point2D(X * L, this.Y * L);
      }
    }

    #endregion

    [ThreadStatic]
    private static readonly Random s_UniformRandom = new Random();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double GenerateUniform(double a, double b)
    {
      var x = s_UniformRandom.NextDouble();
      return a + x * (b - a);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Point2D GenerateUniformPoint()
    {
      return new Point2D(Utils.GenerateUniform(-1, 1), Utils.GenerateUniform(-1, 1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Point2D GenerateNormalPoint(double muX, double muY, double sigma)
    {
      Point2D? sample = null;
      while (!sample.HasValue)
      {
        var p = Utils.GenerateUniformPoint();
        sample = p.ToBoxMuller();
      }

      return new Point2D(sample.Value.X * sigma + muX, sample.Value.Y * sigma + muY);
    }
  }
}
