using System;
using ML.Core.Contracts;

namespace ML.Core.Algorithms
{
  public abstract class KernelAlgorithmBase<TParam> : OrderedMetricAlgorithmBase<TParam>
  {
    private readonly IKernel m_Kernel;

    public KernelAlgorithmBase(ClassifiedSample classifiedSample,
                               IMetric metric,
                               IKernel kernel,
                               TParam pars)
      : base(classifiedSample, metric, pars)
    {
      if (kernel == null)
        throw new ArgumentException("KernelAlgorithmBase.ctor(kernel=null)");

      m_Kernel = kernel;
    }

    public IKernel Kernel { get { return m_Kernel; } }
  }
}
