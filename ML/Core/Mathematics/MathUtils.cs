using System;
using System.Runtime.CompilerServices;

namespace ML.Core.Mathematics
{
  /// <summary>
  /// Utilitary Math functions
  /// </summary>
  public static partial class MathUtils
  {
    public const double ENTROPY_COEFF = 1.44269504089F; // 1/ln(2)

    /// <summary>
    /// Calculates h(z) = -z*log2(z)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double EntropyH(double z)
    {
      return (0.0D <= z && z < double.Epsilon) ? 0.0D : -z*Math.Log(z)*ENTROPY_COEFF;
    }

    /// <summary>
    /// Calculates maximum value within array alog with its index
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ArgMax<T>(T[] array)
      where T : IComparable<T>
    {
      var idx = -1;
      var max = default(T);

      if (array==null) return -1;

      var len = array.Length;
      for (int i=0; i<len; i++)
      {
        var val = array[i];
        if (i==0 || val.CompareTo(max)>0)
        {
          idx = i;
          max = val;
        }
      }

      return idx;
    }

    /// <summary>
    /// Throws if arrays have different lenghts
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void CheckDimensions(double[] p1, double[] p2)
    {
      if (p1.Length != p2.Length)
        throw new MLException("Can not add point with different dimension");
    }
  }
}
