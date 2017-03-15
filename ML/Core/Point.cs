using System;
using System.Linq;
using ML.Contracts;

namespace ML.Core
{
  /// <summary>
  /// Represents a 2D point
  /// </summary>
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

    public override int GetHashCode()
    {
      return X.GetHashCode() ^ Y.GetHashCode();
    }

    public override bool Equals(object obj)
    {
      if (obj==null || !(obj is Point2D)) return false;
      return X.Equals(((Point2D)obj).X) && Y.Equals(((Point2D)obj).Y);
    }
  }
}
