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
    private TParam[][] m_DistributionParameters;


    public NaiveBayesianGeneralAlgorithm(TDistr distribution, double[] classLosses=null)
      : base(classLosses)
    {
      if (distribution == null)
        throw new MLException("NaiveBayesianGeneralAlgorithm.ctor(distribution=null)");

      m_Distribution = distribution;
    }

    public override string Name   { get { return "NPBAYES"; } }

    public TDistr Distribution { get { return m_Distribution; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override ClassScore[] PredictTokens(double[] obj, int cnt)
    {
      var classes = Classes;
      var priors  = PriorProbs;
      var dim     = DataDim;
      var scores  = new List<ClassScore>();

      foreach (var cls in classes)
      {
        var p = priors[cls.Value];
        var ds = m_DistributionParameters[cls.Value];

        for (int i=0; i<dim; i++)
        {
          m_Distribution.Params = ds[i];
          var value = m_Distribution.LogValue(obj[i]);

          p += value;
        }

        scores.Add(new ClassScore(cls, p));
      }

      return scores.OrderByDescending(s => s.Score)
                   .Take(cnt)
                   .ToArray();
    }

    /// <summary>
    /// Estimated proximity of given point to given classes
    /// </summary>
    public override double CalculateClassScore(double[] obj, Class cls)
    {
      var dim = TrainingSample.GetDimension();
      var p   = PriorProbs[cls.Value];
      var ds = m_DistributionParameters[cls.Value];

      foreach (var pData in TrainingSample.Where(d => d.Value.Equals(cls)))
      {
        var data = pData.Key;

        for (int i=0; i<dim; i++)
        {
          m_Distribution.Params = ds[i];
          var value = m_Distribution.LogValue(obj[i]);

          p += value;
        }
      }

      return p;
    }


    protected override void TrainImpl()
    {
      m_DistributionParameters = m_Distribution.FromSample(TrainingSample);
    }
  }
}
