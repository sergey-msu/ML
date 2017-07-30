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
                                        Dictionary<Class, double> classLosses=null)
      : base(kernel, h, classLosses)
    {
    }

    public override string Name   { get { return "NBAYES"; } }



    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var dim = DataDim;
      var cnt = DataCount;
      var classes = DataClasses;
      var pHist = new Dictionary<Class, double>();
      var yHist = new Dictionary<Class, double>();

      foreach (var cls in classes)
      {
        pHist[cls] = 0.0D;
        yHist[cls] = 0.0D;
      }

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in TrainingSample)
        {
          var data = pData.Key;
          var cls  = pData.Value;

          var r = (obj[i] - data[i])/H;
          pHist[cls] += Kernel.Value(r);
        }

        foreach (var cls in classes)
        {
          yHist[cls] += Math.Log(pHist[cls] / (H * ClassHist[cls]));
          pHist[cls] = 0.0D;
        }
      }

      var max = double.MinValue;
      var result = Class.Unknown;
      foreach (var cls in classes)
      {
        double penalty;
        if (ClassLosses == null || !ClassLosses.TryGetValue(cls, out penalty)) penalty = 1;
        var p = yHist[cls] + Math.Log(PriorProbs[cls]*penalty);
        if (p > max)
        {
          max = p;
          result = cls;
        }
      }

      return result;
    }

    /// <summary>
    /// Estimated closeness of given point to given classes
    /// </summary>
    public override double CalculateClassScore(double[] obj, Class cls)
    {
      var dim = DataDim;
      var p = 0.0D;
      var y = 0.0D;
      var my = ClassHist[cls];

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in TrainingSample.Where(d => d.Value.Equals(cls)))
        {
          var data = pData.Key;

          var r = (obj[i] - pData.Key[i])/H;
          p += Kernel.Value(r);
        }

        y += Math.Log(p / (H * my));
        p = 0.0D;
      }

      double penalty;
      if (ClassLosses == null || !ClassLosses.TryGetValue(cls, out penalty)) penalty = 1;
      y += Math.Log(penalty*PriorProbs[cls]);

      return y;
    }
  }
}
