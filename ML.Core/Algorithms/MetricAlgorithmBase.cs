using System;
using System.Collections.Generic;
using ML.Core.Contracts;

namespace ML.Core.Algorithms
{
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

    public IMetric Metric { get { return m_Metric; } }
  }
}
