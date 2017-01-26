using System;
using System.Runtime.CompilerServices;
using ML.Core;

namespace ML.Mathematics
{
  /// <summary>
  /// Unitilitary functions
  /// </summary>
  public static class MathUtils
  {
    [ThreadStatic]
    private static readonly Random s_UniformRandom = new Random();

    /// <summary>
    /// Calculates h(z) = -z log2(z)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double EntropyH(float z)
    {
      const double COEFF = 1.44269504089F; // 1/ln(2)
      return (0.0D <= z && z < double.Epsilon) ? 0.0D : -z*Math.Log(z)*COEFF;
    }

    /// <summary>
    /// Returns uniformly distributed random value
    /// </summary>
    /// <param name="a">Min value</param>
    /// <param name="b">Max value</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double GenerateUniform(double a, double b)
    {
      var x = s_UniformRandom.NextDouble();
      return a + x * (b - a);
    }

    /// <summary>
    /// Returns uniformly distributed in [-1, 1]x[-1, 1] 2D point
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Point2D GenerateUniformPoint()
    {
      return new Point2D(MathUtils.GenerateUniform(-1, 1), MathUtils.GenerateUniform(-1, 1));
    }

    /// <summary>
    /// Returns Box-Muller normally distributed 2D point
    /// </summary>
    /// <param name="muX">Mean X</param>
    /// <param name="muY">Mean Y</param>
    /// <param name="sigma">Sigma</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Point2D GenerateNormalPoint(double muX, double muY, double sigma)
    {
      Point2D? sample = null;
      while (!sample.HasValue)
      {
        var p = MathUtils.GenerateUniformPoint();
        sample = p.ToBoxMuller();
      }

      return new Point2D(sample.Value.X * sigma + muX, sample.Value.Y * sigma + muY);
    }
  }
}
