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

  /// <summary>
  /// Represents a multidimentional point
  /// </summary>
  public struct Point
  {
    private readonly double[] m_SpacePoint;

    public Point(int dim)
    {
      m_SpacePoint = new double[dim];
      Dimension = dim;
    }

    public Point(params double[] point)
    {
      if (point==null || point.Length<=0)
        throw new MLException("Point.ctor(point=null|empty)");

      m_SpacePoint = point.ToArray();
      Dimension = m_SpacePoint.Length;
    }

    /// <summary>
    /// Dimension
    /// </summary>
    public readonly int Dimension;

    /// <summary>
    /// Returns i-th point coordinate value
    /// </summary>
    public double this[int i]
    {
      get { return m_SpacePoint[i]; }
      set { m_SpacePoint[i] = value; }
    }

    #region Overrides

    public override int GetHashCode()
    {
      return m_SpacePoint.GetHashCode();
    }

    public override bool Equals(object obj)
    {
      if (obj==null || !(obj is Point)) return false;
      return this.m_SpacePoint.Equals(((Point)obj).m_SpacePoint);
    }

    public static Point operator +(Point p1, Point p2)
    {
      Point.CheckDimensions(p1, p2);

      var dim = p1.Dimension;
      var point = new double[dim];
      for (int i=0; i<dim; i++)
        point[i] = p1.m_SpacePoint[i] + p2.m_SpacePoint[i];

      return new Point(point);
    }

    public static Point operator -(Point p1, Point p2)
    {
      Point.CheckDimensions(p1, p2);

      var dim = p1.Dimension;
      var point = new double[dim];
      for (int i=0; i<dim; i++)
        point[i] = p1.m_SpacePoint[i] - p2.m_SpacePoint[i];

      return new Point(point);
    }

    public static Point operator *(double c, Point p1)
    {
      var dim = p1.Dimension;
      var point = new double[dim];
      for (int i=0; i<dim; i++)
        point[i] = c * p1.m_SpacePoint[i];

      return new Point(point);
    }

    public static bool operator ==(Point p1, Point p2)
    {
      return p1.Equals(p2);
    }

    public static bool operator !=(Point p1, Point p2)
    {
      return !p1.Equals(p2);
    }

    public static void CheckDimensions(Point p1, Point p2)
    {
      if (p1.Dimension != p2.Dimension)
        throw new MLException("Can not add point with different dimension");
    }

    public static implicit operator double[](Point x)
    {
       return x.m_SpacePoint;
    }

    #endregion
  }
}
