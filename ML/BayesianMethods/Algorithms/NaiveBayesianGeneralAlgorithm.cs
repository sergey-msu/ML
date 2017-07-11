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
    private readonly Dictionary<Class, double> m_ClassLosses;
    private Dictionary<ClassFeatureKey, TParam> m_DistributionParameters;


    public NaiveBayesianGeneralAlgorithm(TDistr distribution, Dictionary<Class, double> classLosses=null)
      : base(classLosses)
    {
      if (distribution == null)
        throw new MLException("NaiveBayesianGeneralAlgorithm.ctor(distribution=null)");

      m_Distribution = distribution;
      m_ClassLosses = classLosses;
    }

    public override string ID { get { return "NPBAYES"; } }
    public override string Name { get { return "Naive Bayesian parametric classification"; } }

    public TDistr Distribution { get { return m_Distribution; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var classes = TrainingSample.CachedClasses;
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
          p += Math.Log(m_Distribution.Value(obj[i]));
        }

        var ly = (m_ClassLosses == null) ? 1.0D : m_ClassLosses[cls];
        p += Math.Log(ClassHist[cls]*ly / cnt);

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
      var l   = 0;

      foreach (var pData in TrainingSample.Where(d => d.Value.Equals(cls)))
      {
        var data = pData.Key;
        l += 1;

        for (int i=0; i<dim; i++)
        {
          var key = new ClassFeatureKey(cls, i);
          m_Distribution.Params = m_DistributionParameters[key];
          p += Math.Log(m_Distribution.Value(data[i]));
        }
      }

      var ly = (m_ClassLosses == null) ? 1.0D : m_ClassLosses[cls];
      p += Math.Log(l*ly / cnt);

      return p;
    }


    protected override void TrainImpl()
    {
      m_DistributionParameters = m_Distribution.MaximumLikelihood(TrainingSample);
    }
  }
}
