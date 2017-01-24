using System;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.Core.Algorithms
{
  /// <summary>
  /// Nearest K Neighbours Algorithm
  /// </summary>
  public sealed class NearestKNeighboursAlgorithm : OrderedMetricAlgorithmBase<NearestKNeighboursAlgorithm.Params>
  {
    #region Inner

    public class Params
    {
      public Params(int k)
      {
        if (k <= 0)
          throw new ArgumentException("NearestKNeighboursAlgorithm.Params.ctor(k<=0)");

        K = k;
      }

      public readonly int K;
    }

    #endregion

    public NearestKNeighboursAlgorithm(ClassifiedSample classifiedSample,
                                       IMetric metric,
                                       Params pars)
      : base(classifiedSample, metric, pars)
    {
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public override string ID { get { return "NNK"; } }

    /// <summary>
    /// Algorithm name
    /// </summary>
    public override string Name { get { return "Nearest K Neighbour(s)"; } }

    /// <summary>
    /// Calculate 'weight' - a contribution of training point (i-th from ordered training sample)
    /// to closeness of test point to its class
    /// </summary>
    /// <param name="i">Point number in ordered training sample</param>
    /// <param name="x">Test point</param>
    /// <param name="orderedSample">Ordered training sample</param>
    protected override float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample)
    {
      return (i < Parameters.K) ? 1 : 0;
    }
  }
}
