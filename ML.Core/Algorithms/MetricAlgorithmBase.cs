using System;
using ML.Core.Contracts;

namespace ML.Core.Algorithms
{
  /// <summary>
  /// Base class for algorithm supplied with some spacial metric
  /// </summary>
  public abstract class MetricAlgorithmBase<TParam> : AlgorithmBase<TParam>
  {
    public readonly IMetric m_Metric;

    protected MetricAlgorithmBase(ClassifiedSample classifiedSample,
                                  IMetric metric,
                                  TParam pars)
      : base(classifiedSample, pars)
    {
      if (metric == null)
        throw new ArgumentException("MetricAlgorithmBase.ctor(metric=null)");

      m_Metric = metric;
    }

    /// <summary>
    /// Space metric
    /// </summary>
    public IMetric Metric { get { return m_Metric; } }
  }
}
