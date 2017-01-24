using System;
using System.Collections.Generic;
using ML.Contracts;


namespace ML.Core.Algorithms
{
  /// <summary>
  /// Nearest Neighbour Algorithm
  /// </summary>
  public sealed class NearestNeighbourAlgorithm : OrderedMetricAlgorithmBase<NearestNeighbourAlgorithm.Params>
  {
    #region Inner

    public class Params {}

    #endregion

    public NearestNeighbourAlgorithm(ClassifiedSample classifiedSample, IMetric metric, Params pars=null)
      : base(classifiedSample, metric, pars)
    {
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public override string ID { get { return "NN"; } }

    /// <summary>
    /// Algorithm name
    /// </summary>
    public override string Name { get { return "Nearest K Neighbours"; } }

    /// <summary>
    /// Calculate 'weight' - a contribution of training point (i-th from ordered training sample)
    /// to closeness of test point to its class
    /// </summary>
    /// <param name="i">Point number in ordered training sample</param>
    /// <param name="x">Test point</param>
    /// <param name="orderedSample">Ordered training sample</param>
    protected override float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample)
    {
      return i==0 ? 1 : 0;
    }
  }
}
