using System;

namespace ML.Core.Mathematics
{
  /// <summary>
  /// Utilitary Math functions
  /// </summary>
  public static partial class MatrixOps
  {
    /// <summary>
    /// Given input square, symmetric, and positive definite matrix A, returns the lower Cholesky Factor, L
    /// wich is defined by: L*L^T = A.
    /// </summary>
    public static double[,] CholeskyFactor(double[,] A)
    {
      var n = (int)Math.Sqrt(A.Length);
      var L = new double[n, n];

      for (int i = 0; i < n;  i++)
      for (int j = 0; j <= i; j++)
      {
        var sum = 0.0D;

        if (j == i)
        {
          for (int k = 0; k < j; k++)
          {
            var ljk = L[j, k];
            sum += ljk*ljk;
          }
          L[j, j] = Math.Sqrt(A[j, j] - sum);
        }
        else
        {
          for (int k = 0; k < j; k++)
            sum += L[i, k]*L[j, k];
          L[i, j] = (A[i, j] - sum)/L[j, j];
        }
      }

      return L;
    }

    public static double[,] LowerTriangelInverce(double[,] L)
    {
      var n = L.GetLength(0);
      var R = new double[n,n];

      for (var k=0; k<n; k++)
      {
        R[k,k] = 1.0D/L[k,k];
        for (var i=k+1; i<n; i++)
        {
          var s = 0.0D;
          for (var j=k; j<i; j++)
            s += L[i, j]*R[j, k];
          R[i,k] = -s/L[i,i];
        }
      }

      return R;
    }

    public static double[,] Mult(double[,] A, double[,] B)
    {
      var m = A.GetLength(0);
      var n = A.GetLength(1);
      var k = B.GetLength(1);
      var C = new double[m,k];

      for (int i=0; i<m; i++)
      for (int j=0; j<k; j++)
      {
        var res = 0.0D;
        for (int l=0; l<n; l++)
          res += A[i,l]*B[l,j];

        C[i, j] = res;
      }

      return C;
    }

    public static double[,] Transpose(double[,] A)
    {
      var m = A.GetLength(0);
      var n = A.GetLength(0);
      var T = new double[n, m];

      for (int i=0; i<m; i++)
      for (int j=0; j<n; j++)
      {
        T[j, i] = A[i, j];
      }

      return T;
    }
  }
}
