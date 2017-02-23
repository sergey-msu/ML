using System;

namespace ML.Core.Mathematics
{
  /// <summary>
  /// Random unitilitary container
  /// </summary>
  public class RandomGenerator
  {
    private readonly Random m_Random = new Random();

    public RandomGenerator(int seed=0)
    {
      m_Random = new Random(seed);
    }

    /// <summary>
    /// Returns uniformly distributed random value
    /// </summary>
    /// <param name="a">Min value</param>
    /// <param name="b">Max value</param>
    public double GenerateUniform(double a, double b)
    {
      var x = m_Random.NextDouble();
      return a + x * (b - a);
    }

    /// <summary>
    /// Returns uniformly distributed in [-1, 1]x[-1, 1] 2D point
    /// </summary>
    public Point2D GenerateUniformPoint()
    {
      return new Point2D(GenerateUniform(-1, 1), GenerateUniform(-1, 1));
    }

    /// <summary>
    /// Returns Box-Muller normally distributed 2D point
    /// </summary>
    /// <param name="muX">Mean X</param>
    /// <param name="muY">Mean Y</param>
    /// <param name="sigma">Sigma</param>
    public Point2D GenerateNormalPoint(double muX, double muY, double sigma)
    {
      Point2D? sample = null;
      while (!sample.HasValue)
      {
        var p = GenerateUniformPoint();
        sample = p.ToBoxMuller();
      }

      return new Point2D(sample.Value.X * sigma + muX, sample.Value.Y * sigma + muY);
    }
  }
}
