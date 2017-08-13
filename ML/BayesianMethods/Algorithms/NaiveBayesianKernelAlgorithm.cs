using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Utils;

namespace ML.BayesianMethods.Algorithms
{
  /// <summary>
  /// Naive Bayesian non-parametric classification algorithm.
  ///
  /// a(x) = argmax[ ly*P(y)*p(x|y) ]
  /// where p(x|y) = PROD( p(xj|y), j=1..n) = PROD( 1/m_y*SUMM( K((xj-xji)/h)/h, i=1..m, y_i=y), j=1..n),
  /// xj  - j-th feature of x
  /// xji - j-th feature of i-th training object x_i
  /// ly  - penalty for error on object of class y
  /// m   - number of training objects
  /// m_y - number of training objects in of class y
  /// n   - feature space dimension
  ///
  /// Deals with a probability distributions on classes (not to be confused with Bayesian learning, where probability distributions are considered on algorithm parameters)
  /// in a special case of independent (as random variables) features.
  /// If class multiplicative penalties are absent, the algorithm is non-parametric Parzen window implementation of Maximum posterior probability (MAP) classification
  /// </summary>
  public class NaiveBayesianKernelAlgorithm : BayesianKernelAlgorithmBase
  {
    public NaiveBayesianKernelAlgorithm(IKernel kernel,
                                        double h = 1,
                                        double[] classLosses=null)
      : base(kernel, h, classLosses)
    {
    }


    public override string Name { get { return "NBAYES"; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override ClassScore[] PredictTokens(double[] obj, int cnt)
    {
      var dim  = DataDim;
      var dcnt = DataCount;
      var classes = Classes;
      var useMin = UseKernelMinValue;
      var min    = KernelMinValue;
      var pHist  = new double[classes.Length];
      var yHist  = new double[classes.Length];
      var scores = new List<ClassScore>();

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in TrainingSample)
        {
          var data = pData.Key;
          var cls  = pData.Value;
          var r = (obj[i] - data[i])/H;

          pHist[cls.Value] += Kernel.Value(r);
        }

        foreach (var cls in classes)
        {
          var p = pHist[cls.Value] / (H * ClassHist[cls.Value]);
          if (Math.Abs(p)<min && useMin) p = min;

          yHist[cls.Value] += Math.Log(p);
          pHist[cls.Value] = 0.0D;
        }
      }

      foreach (var cls in classes)
      {
        var p = yHist[cls.Value] + PriorProbs[cls.Value];
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
      var dim = DataDim;
      var p = 0.0D;
      var y = 0.0D;
      var my = ClassHist[cls.Value];
      var useMin = UseKernelMinValue;
      var min = KernelMinValue;

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in TrainingSample.Where(d => d.Value.Equals(cls)))
        {
          var data = pData.Key;
          var r = (obj[i] - pData.Key[i])/H;
          p += Kernel.Value(r);
        }
        p = p / (H * my);
        if (Math.Abs(p)<min && useMin) p = min;
        y += Math.Log(p);
        p = 0.0D;
      }

      y += PriorProbs[cls.Value];

      return y;
    }
  }
}
