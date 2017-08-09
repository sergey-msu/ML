using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.BayesianMethods.Algorithms
{
  /// <summary>
  /// Bayesian non-parametric classification algorithm with product-like multidimensional kernel.
  ///
  /// a(x) = argmax[ ly*P(y)*p(x|y) ]
  /// where p(x|y) = 1/m*SUMM( PROD( K((xj-xji)/h)/h, j=1..n), i=1..m),
  /// xj  - j-th feature of x
  /// xji - j-th feature of i-th training object x_i
  /// ly  - penalty for error on object of class y
  /// m   - number of training objects
  /// n   - feature space dimension
  ///
  /// Deals with a probability distributions on classes (not to be confused with Bayesian learning, where probability distributions are considered on algorithm parameters).
  /// If class multiplicative penalties are absent, the algorithm is non-parametric Parzen window implementation of maximum posterior probability (MAP) classification.
  /// Key idea is to model multidimensional distributions as products of Parzen approximation of all features
  /// </summary>
  public class BayesianKernelAlgorithm : BayesianKernelAlgorithmBase
  {
    private readonly double[] m_Hs;

    public BayesianKernelAlgorithm(IKernel kernel,
                                   double h = 1,
                                   Dictionary<Class, double> classLosses=null,
                                   double[] hs = null)
      : base(kernel, h, classLosses)
    {
      m_Hs = hs;
    }


    public override string Name { get { return "PKBAYES"; } }

    /// <summary>
    /// Window widths
    /// </summary>
    public double[] Hs { get { return m_Hs; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override ClassScore[] PredictTokens(double[] obj, int cnt)
    {
      var dim = DataDim;
      var classes = DataClasses;
      var useMin = UseKernelMinValue;
      var min = KernelMinValue;
      var pHist = new Dictionary<Class, double>();

      foreach (var pData in TrainingSample)
      {
        var data = pData.Key;
        var cls  = pData.Value;

        var p = 0.0D;
        for (int i=0; i<dim; i++)
        {
          var h = (m_Hs != null ) ? m_Hs[i] : H;
          var r = (obj[i] - data[i])/h;
          var v = Kernel.Value(r);
          if (Math.Abs(v)<min && useMin) v = min;

          p += Math.Log(v);
        }

        if (!pHist.ContainsKey(cls)) pHist[cls] = p;
        else pHist[cls] += p;
      }

      var scores = new List<ClassScore>();
      foreach (var cls in classes)
      {
        var p = pHist[cls] + PriorProbs[cls];
        scores.Add(new ClassScore(cls, p));
      }

      return scores.OrderByDescending(s => s.Score)
                   .Take(cnt)
                   .ToArray();
    }

    /// <summary>
    /// Estimated closeness of given point to given classes
    /// </summary>
    public override double CalculateClassScore(double[] obj, Class cls)
    {
      var score = 0.0D;
      var dim = DataDim;
      var useMin = UseKernelMinValue;
      var min = KernelMinValue;

      foreach (var pData in TrainingSample.Where(d => d.Value.Equals(cls)))
      {
        var data = pData.Key;

        var p = 0.0D;
        for (int i=0; i<dim; i++)
        {
          var h = (m_Hs != null ) ? m_Hs[i] : H;
          var r = (obj[i] - data[i])/h;
          var v = Kernel.Value(r)/h;
          if (Math.Abs(v)<min && useMin) v = min;
          p += Math.Log(v);
        }

        score += p;
      }

      score += PriorProbs[cls];

      return score;
    }
  }
}
