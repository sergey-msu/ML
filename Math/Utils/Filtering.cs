using System;
using System.Collections.Generic;

namespace Math.Utils
{
  public static class Filtering
  {
    /// <summary>
    /// Applies simple Kalman filter
    /// </summary>
    /// <param name="input">Input data array</param>
    /// <param name="output">Filtered data array</param>
    /// <param name="r">Squared sigma of eta</param>
    /// <param name="q">Squared sigma of xi</param>
    public static void Kalman(double[] input, double[] output, double r, double q, bool forward = true)
    {
      var p = r;
      var k = 0.0D;
      var n = input.Length;

      if (forward)
      {
        var z = input[0];
        output[0] = z;

        for (int i=1; i<n; i++)
        {
          k = (p + q)/(p + q + r);
          p = r*k;
          z = z + (input[i] - z)*k;

          output[i] = z;
        }
      }
      else
      {
        var z = input[n-1];
        output[n-1] = z;

        for (int i=n-2; i>=0; i--)
        {
          k = (p + q)/(p + q + r);
          p = r*k;
          z = z + (input[i] - z)*k;

          output[i] = z;
        }
      }
    }

    public static void Smooth(double[] input, double[] smooth, double[] output)
    {
      var n = input.Length;
      var sn = smooth.Length;
      var sn2 = (sn - 1)/2;

      for (int i=0; i<n; i++)
      {
        var s = 0.0D;

        for (int j=0; j<sn; j++)
        {
          var ii = i + j - sn2;
          if (ii<0 || ii>=n) continue;
          s += input[ii]*smooth[j];
        }

        output[i] = s;
      }
    }
  }
}
