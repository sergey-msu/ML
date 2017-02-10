using System;
using System.Runtime.CompilerServices;

namespace ML.Core.Mathematics
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

    /// <summary>
    /// Calculates maximum value within array alog with its index
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void CalcMax(double[] array, out int idx, out double max)
    {
      idx = -1;
      max = double.MinValue;

      if (array==null) return;

      for (int i=0; i<array.Length; i++)
      {
        var val = array[i];
        if (idx<0 || val > max)
        {
          idx = i;
          max = val;
        }
      }
    }

  }
}
