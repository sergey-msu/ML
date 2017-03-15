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

    /// <summary>
    /// Calculates maximum value within array alog with its index
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void CalcMax(double[,,] array, out int iidx, out int jidx, out int kidx, out double max)
    {
      iidx = -1;
      jidx = -1;
      kidx = -1;
      max = double.MinValue;

      if (array==null) return;

      int imax = array.GetLength(0);
      int jmax = array.GetLength(1);
      int kmax = array.GetLength(2);

      for (int i=0; i<imax; i++)
      for (int j=0; j<jmax; j++)
      for (int k=0; k<kmax; k++)
      {
        var val = array[i,j,k];
        if (iidx<0 || jidx<0 || kidx<0 || val > max)
        {
          iidx = i;
          jidx = j;
          kidx = k;
          max = val;
        }
      }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void CheckDimensions(double[] p1, double[] p2)
    {
      if (p1.Length != p2.Length)
        throw new MLException("Can not add point with different dimension");
    }

  }
}
