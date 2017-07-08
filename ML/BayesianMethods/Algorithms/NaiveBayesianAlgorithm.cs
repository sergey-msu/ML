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
  /// Deals with a probability distributions on classes (not to be confused with Bayesian learning, where probability distributions are considered on algorithm parameters)
  /// in a special case of independent (as random variables) features.
  /// If class multiplicative penalties are absent, the algorithm is non-parametric Parzen window implementation of Maximum posterior probability (MAP) classification
  /// </summary>
  public class NaiveBayesianAlgorithm : ClassificationAlgorithmBase<double[]>, IKernelAlgorithm<double[]>
  {
    private readonly IKernel m_Kernel;
    private readonly Dictionary<Class, double> m_ClassLosses;
    private double m_H;

    public NaiveBayesianAlgorithm(IKernel kernel,
                                  double h = 1,
                                  Dictionary<Class, double> classLosses=null)
    {
      if (kernel == null)
        throw new MLException("BayesianAlgorithm.ctor(kernel=null)");

      m_Kernel = kernel;
      m_ClassLosses = classLosses;
      H = h;
    }

    public override string ID { get { return "NBAYES"; } }

    public override string Name { get { return "Naive Bayesian non-parametric classification"; } }

    /// <summary>
    /// Kernel function
    /// </summary>
    public IKernel Kernel { get { return m_Kernel; } }

    /// <summary>
    /// Additional multiplicative penalty to wrong object classification.
    /// If null, all class penalties dafault to 1 (no special effect on classification - pure MAP classification)
    /// </summary>
    public Dictionary<Class, double> ClassLosses { get { return m_ClassLosses; } }

    /// <summary>
    /// Window width
    /// </summary>
    public double H
    {
     get { return m_H; }
     set
     {
       if (value <= double.Epsilon)
         throw new MLException("BayesianAlgorithm.H(value<=0)");

       m_H = value;
     }
    }


    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var dim     = TrainingSample.GetDimension();
      var classes = TrainingSample.CachedClasses;

      var lHist = new Dictionary<Class, int>();
      var pHist = new Dictionary<Class, double>();
      var yHist = new Dictionary<Class, double>();
      foreach (var cls in classes)
      {
        lHist[cls] = 0;
        pHist[cls] = 0.0D;
        yHist[cls] = 0.0D;
      }

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in TrainingSample)
        {
          var data = pData.Key;
          var cls  = pData.Value;

          if (i==0) lHist[cls] += 1;

          var r = (obj[i] - data[i])/m_H;
          pHist[cls] += Kernel.Value(r);
        }

        foreach (var cls in classes)
        {
          yHist[cls] += Math.Log(pHist[cls] / (m_H * lHist[cls]));
          pHist[cls] = 0.0D;
        }
      }

      foreach (var cls in classes)
      {
        var ly = (m_ClassLosses == null) ? 1.0D : m_ClassLosses[cls];
        yHist[cls] += Math.Log(lHist[cls]*ly / TrainingSample.Count);
      }

      var max = double.MinValue;
      Class result = null;
      foreach (var cls in classes)
      {
        var prob = yHist[cls];
        if (prob > max)
        {
          max = prob;
          result = cls;
        }
      }

      return result;
    }

    /// <summary>
    /// Estimated closeness of given point to given classes
    /// </summary>
    public double CalculateClassScore(double[] obj, Class cls)
    {
      var dim = TrainingSample.GetDimension();
      var my = 0;
      var p = 0.0D;
      var y = 0.0D;

      for (int i=0; i<dim; i++)
      {
        foreach (var pData in TrainingSample.Where(d => d.Value.Equals(cls)))
        {
          var data = pData.Key;

          if (i==0) my += 1;

          var r = (obj[i] - pData.Key[i])/m_H;
          p += Kernel.Value(r);
        }

        y += Math.Log(p / (m_H * my));
        p = 0.0D;
      }

      double penalty;
      if (m_ClassLosses != null && m_ClassLosses.TryGetValue(cls, out penalty))
        y += Math.Log(my*penalty / TrainingSample.Count);

      return y;
    }

    protected override void DoTrain()
    {
      // Nonparametric Bayesian methods are not trainable by default
    }
  }
}
