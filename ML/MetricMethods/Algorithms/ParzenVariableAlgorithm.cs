using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;

namespace ML.MetricMethods.Algorithms
{
  /// <summary>
  /// Parzen Variable Window Algorithm
  /// </summary>
  public sealed class ParzenVariableAlgorithm : OrderedMetricAlgorithmBase<double[]>, IKernelAlgorithm<double[]>
  {
    private readonly IKernel m_Kernel;
    private int m_K;

    public ParzenVariableAlgorithm(IMetric<double[]> metric,
                                   IKernel kernel,
                                   int k)
      : base(metric)
    {
      if (kernel == null)
        throw new MLException("KernelAlgorithmBase.ctor(kernel=null)");

      m_Kernel = kernel;
      K = k;
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public override string Name { get { return "PVW"; } }

    public IKernel Kernel { get { return m_Kernel; } }

    /// <summary>
    /// Neighbour count
    /// </summary>
    public int K
    {
      get { return m_K; }
      set
      {
        if (value <= 0)
          throw new MLException("ParzenVariableAlgorithm.K(value<=0)");

        m_K = value;
      }
    }

    public double H
    {
      get { return m_K; }
      set { throw new NotSupportedException("Use K parameter instead"); }
    }

    /// <summary>
    /// Calculate 'weight' - a contribution of training point (i-th from ordered training sample)
    /// to closeness of test point to its class
    /// </summary>
    /// <param name="i">Point number in ordered training sample</param>
    /// <param name="x">Test point</param>
    /// <param name="orderedSample">Ordered training sample</param>
    protected override double CalculateWeight(int i, double[] x, Dictionary<double[], double> orderedSample)
    {
      var r = orderedSample.ElementAt(i).Value / orderedSample.ElementAt(m_K+1).Value;
      return Kernel.Value(r);
    }
  }
}
