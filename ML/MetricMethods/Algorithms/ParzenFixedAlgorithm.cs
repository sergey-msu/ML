using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;

namespace ML.MetricMethods.Algorithms
{
  /// <summary>
  /// Parzen Fixed Window Algorithm
  /// </summary>
  public sealed class ParzenFixedAlgorithm : OrderedMetricAlgorithmBase<double[]>, IKernelAlgorithm<double[]>
  {
    private readonly IKernel m_Kernel;
    private double m_H;

    public ParzenFixedAlgorithm(IMetric<double[]> metric,
                                IKernel kernel,
                                double h = 1)
      : base(metric)
    {
      if (kernel == null)
        throw new MLException("PotentialAlgorithm.ctor(kernel=null)");

      m_Kernel = kernel;
      H = h;
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public override string ID { get { return "PFW"; } }

    /// <summary>
    /// Algorithm name
    /// </summary>
    public override string Name { get { return "Parzen Fixed Width Window"; } }

    /// <summary>
    /// Window width
    /// </summary>
    public double H
    {
      get { return m_H; }
      set
      {
        if (value <= double.Epsilon)
          throw new MLException("ParzenFixedAlgorithm.H(value<=0)");

        m_H = value;
      }
    }

    public IKernel Kernel { get { return m_Kernel; } }


    /// <summary>
    /// Calculate 'weight' - a contribution of training point (i-th from ordered training sample)
    /// to closeness of test point to its class
    /// </summary>
    /// <param name="i">Point number in ordered training sample</param>
    /// <param name="x">Test point</param>
    /// <param name="orderedSample">Ordered training sample</param>
    protected override double CalculateWeight(int i, double[] x, Dictionary<double[], double> orderedSample)
    {
      var r = orderedSample.ElementAt(i).Value / m_H;
      return Kernel.Value(r);
    }
  }
}
