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
    /// <summary>
    /// Calculates h(z) = -z log2(z)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double EntropyH(double z)
    {
      const double COEFF = 1.44269504089F; // 1/ln(2)
      return (0.0D <= z && z < double.Epsilon) ? 0.0D : -z*Math.Log(z)*COEFF;
    }

  }
}
