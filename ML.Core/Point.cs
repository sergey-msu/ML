using System;
using System.Linq;

namespace ML.Core
{
  public struct Point
  {
    private readonly float[] m_SpacePoint;

    public Point(int dimension)
    {
      m_SpacePoint = new float[dimension];
    }

    public Point(float[] point)
    {
      if (point==null || point.Length<=0)
        throw new ArgumentException("Point.ctor(point=null|empty)");

      m_SpacePoint = point.ToArray();
    }

    public int Dimension
    {
      get { return m_SpacePoint.Length; }
    }

    public float this[int i]
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
      var point = new float[dim];
      for (int i=0; i<dim; i++)
        point[i] = p1.m_SpacePoint[i] + p2.m_SpacePoint[i];

      return new Point(point);
    }

    public static Point operator -(Point p1, Point p2)
    {
      Point.CheckDimensions(p1, p2);

      var dim = p1.Dimension;
      var point = new float[dim];
      for (int i=0; i<dim; i++)
        point[i] = p1.m_SpacePoint[i] - p2.m_SpacePoint[i];

      return new Point(point);
    }

    public static Point operator *(float c, Point p1)
    {
      var dim = p1.Dimension;
      var point = new float[dim];
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
        throw new InvalidOperationException("Can not add point with different dimension");
    }

    #endregion
  }
}
