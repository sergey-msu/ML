using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;

namespace ML.MetricalMethods.Algorithms
{
  /// <summary>
  /// Parzen Variable Window Algorithm
  /// </summary>
  public sealed class ParzenVariableAlgorithm : KernelAlgorithmBase
  {
    private int m_K;

    public ParzenVariableAlgorithm(ClassifiedSample classifiedSample,
                                   IMetric metric,
                                   IFunction kernel,
                                   int k)
      : base(classifiedSample, metric, kernel)
    {
      K = k;
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public override string ID { get { return "PVW"; } }

    /// <summary>
    /// Algorithm name
    /// </summary>
    public override string Name { get { return "Parzen Window of Variable Width"; } }

    /// <summary>
    /// Neighbour count
    /// </summary>
    public int K
    {
      get { return m_K; }
      set
      {
        if (value <= 0)
          throw new MLException("NearestKNeighboursAlgorithm.K(value<=0)");

        m_K = value;
      }
    }

    /// <summary>
    /// Calculate 'weight' - a contribution of training point (i-th from ordered training sample)
    /// to closeness of test point to its class
    /// </summary>
    /// <param name="i">Point number in ordered training sample</param>
    /// <param name="x">Test point</param>
    /// <param name="orderedSample">Ordered training sample</param>
    protected override double CalculateWeight(int i, Point x, Dictionary<Point, double> orderedSample)
    {
      var r = orderedSample.ElementAt(i).Value / orderedSample.ElementAt(m_K+1).Value;
      return Kernel.Value(r);
    }
  }
}
