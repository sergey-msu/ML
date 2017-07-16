using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Utils;
using ML.Core.Distributions;

namespace ML.BayesianMethods.Algorithms
{
  /// <summary>
  /// Naive Bayesian parametric classification algorithm with injectable likelihood distribution functions.
  ///
  /// a(x) = argmax[ ly*P(y)*p(x|y) ]
  /// where p(x|y) = PROD( p(xj|y), j=1..n),
  /// xj  - j-th feature of x
  /// ly  - penalty for error on object of class y
  /// n   - feature space dimension
  ///
  /// Deals with a probability distributions on classes (not to be confused with Bayesian learning, where probability distributions are considered on algorithm parameters)
  /// in a special case of independent (as random variables) features.
  /// If class multiplicative penalties are absent, the algorithm is the implementation of Maximum posterior probability (MAP) classification
  /// </summary>
  public class NaiveBayesianGeneralAlgorithm<TDistr, TParam> : BayesianAlgorithmBase
    where TDistr : IDistribution<TParam>
    where TParam : IDistributionParameters
  {
    private readonly TDistr m_Distribution;
    private Dictionary<ClassFeatureKey, TParam> m_DistributionParameters;


    public NaiveBayesianGeneralAlgorithm(TDistr distribution, Dictionary<Class, double> classLosses=null)
      : base(classLosses)
    {
      if (distribution == null)
        throw new MLException("NaiveBayesianGeneralAlgorithm.ctor(distribution=null)");

      m_Distribution = distribution;
    }

    public override string ID   { get { return "NPBAYES"; } }
    public override string Name { get { return "Naive Bayesian parametric classification"; } }

    public TDistr Distribution { get { return m_Distribution; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var classes = DataClasses;
      var dim     = DataDim;
      var cnt     = DataCount;
      var max     = double.MinValue;
      var result  = Class.Unknown;

      foreach (var cls in classes)
      {
        var p = 0.0D;

        for (int i=0; i<dim; i++)
        {
          var key = new ClassFeatureKey(cls, i);
          m_Distribution.Params = m_DistributionParameters[key];
          var value = m_Distribution.Value(obj[i]);
          if (double.IsInfinity(value) || double.IsNaN(value)) return Class.Unknown;

          p += Math.Log(value);
        }

        var ly = (ClassLosses == null) ? 1.0D : ClassLosses[cls];
        p += Math.Log(ly*PriorProbs[cls]);

        if (p > max)
        {
          max = p;
          result = cls;
        }
      }

      return result;
    }

    /// <summary>
    /// Estimated proximity of given point to given classes
    /// </summary>
    public override double CalculateClassScore(double[] obj, Class cls)
    {
      var dim = TrainingSample.GetDimension();
      var cnt = TrainingSample.Count;
      var p   = 0.0D;

      foreach (var pData in TrainingSample.Where(d => d.Value.Equals(cls)))
      {
        var data = pData.Key;

        for (int i=0; i<dim; i++)
        {
          var key = new ClassFeatureKey(cls, i);
          m_Distribution.Params = m_DistributionParameters[key];
          var value = m_Distribution.Value(obj[i]);
          if (double.IsInfinity(value) || double.IsNaN(value)) return double.NaN;

          p += Math.Log(value);
        }
      }

      var ly = (ClassLosses == null) ? 1.0D : ClassLosses[cls];
      p += Math.Log(ly*PriorProbs[cls]);

      return p;
    }


    protected override void TrainImpl()
    {
      m_DistributionParameters = m_Distribution.FromSample(TrainingSample);
    }
  }
}
