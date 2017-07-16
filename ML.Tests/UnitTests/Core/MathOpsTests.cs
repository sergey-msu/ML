using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.Mathematics;
using ML.Utils;

namespace ML.Tests.UnitTests.Core
{
  [TestClass]
  public class MathOpsTests : TestBase
  {
    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    [TestMethod]
    public void CholeskyDecomposition()
    {
      // arrange
      var random = new Random();
      var n = 100;

      var A = new double[n,n];
      var I = new double[n,n];

      for (int i=0; i<n;  i++)
      for (int j=0; j<=i; j++)
      {
        if (i==j)
        {
          A[i,j] = 10;
          I[i,j] = 1;
        }
        else
        {
          var val = random.NextDouble();
          A[i,j] = val;
          A[j,i] = val;
        }
      }

      // act

      var L  = MatrixOps.CholeskyFactor(A);
      var IL = MatrixOps.LowerTriangelInverce(L);
      var IA = MatrixOps.Mult(MatrixOps.Transpose(IL), IL);

      var X = MatrixOps.Mult(L, MatrixOps.Transpose(L));
      Assert.IsTrue(matrixEquals(A, X));

      var Y = MatrixOps.Mult(A, IA);
      Assert.IsTrue(matrixEquals(Y, I));
    }


    private bool matrixEquals(double[,] A, double[,] B, double eps = 0.000000000001D)
    {
      var m = A.GetLength(0);
      if (m != B.GetLength(0)) return false;

      var n = A.GetLength(1);
      if (n != B.GetLength(1)) return false;

      for (int i=0; i<m; i++)
      for (int j=0; j<m; j++)
        if (Math.Abs(A[i,j] - B[i,j]) >= eps) return false;

      return true;
    }

  }
}
