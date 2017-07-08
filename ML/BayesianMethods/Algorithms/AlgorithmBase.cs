using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.BayesianMethods.Algorithms
{
  /// <summary>
  /// Naive Bayesian non-parametric classification algorithm.
  /// Deals with a probability distributions on classes (not to be confused with Bayesian learning, where probability distributions are considered on algorithm parameters)
  /// in a special case of independent (as random variables) features.
  /// If class multiplicative penalties are absent, the algorithm is non-parametric Parzen window implementation of Maximum posterior probability (MAP) classification
  /// </summary>
  public abstract class BayesianAlgorithmBase : ClassificationAlgorithmBase<double[]>, IKernelAlgorithm<double[]>
  {
    private readonly IKernel m_Kernel;
    private readonly Dictionary<Class, double> m_ClassLosses;
    private double m_H;

    public BayesianAlgorithmBase(IKernel kernel, double h, Dictionary<Class, double> classLosses=null)
    {
      if (kernel == null)
        throw new MLException("BayesianAlgorithm.ctor(kernel=null)");

      m_Kernel = kernel;
      m_ClassLosses = classLosses;
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
      return null;
    }

    /// <summary>
    /// Estimated closeness of given point to given classes
    /// </summary>
    public double CalculateClassScore(double[] obj, Class cls)
    {
      var score = 0.0D;

      return score;
    }


    protected override void DoTrain()
    {
      // Nonparametric Bayesian methods are not trainable by default
    }
  }
}
