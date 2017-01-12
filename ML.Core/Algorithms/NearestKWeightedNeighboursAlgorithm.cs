using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core.Contracts;

namespace ML.Core.Algorithms
{
  /// <summary>
  /// Nearest K Neighbours with Weights Algorithm
  /// </summary>
  public sealed class NearestKWeightedNeighboursAlgorithm : OrderedMetricAlgorithmBase<NearestKWeightedNeighboursAlgorithm.Params>
  {
    #region Inner

    public class Params
    {
      public Params(int k, float[] weights=null)
      {
        if (k <= 0)
          throw new ArgumentException("NearestKNeighboursAlgorithm.Params.ctor(k<=0)");

        if (weights==null)
        {
          weights = new float[k];
          for (int i=0; i<k; i++)
            weights[i] = 1;
        }
        else
        {
          weights = weights.ToArray();
        }

        if (k != weights.Length)
          throw new ArgumentException("NearestKNeighboursAlgorithm.Params.ctor(k<>weights.length)");

        K = k;
        Weights = weights;
      }

      public readonly int K;
      public readonly float[] Weights;
    }

    #endregion

    public NearestKWeightedNeighboursAlgorithm(ClassifiedSample classifiedSample,
                                               IMetric metric,
                                               Params pars)
      : base(classifiedSample, metric, pars)
    {
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public override string ID { get { return "WNNK"; } }

    /// <summary>
    /// Algorithm name
    /// </summary>
    public override string Name { get { return "Weighted Nearest K Neighbours"; } }

    /// <summary>
    /// Calculate 'weight' - a contribution of training point (i-th from ordered training sample)
    /// to closeness of test point to its class
    /// </summary>
    /// <param name="i">Point number in ordered training sample</param>
    /// <param name="x">Test point</param>
    /// <param name="orderedSample">Ordered training sample</param>
    protected override float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample)
    {
      return i < Parameters.K ? Parameters.Weights[i] : 0;
    }
  }
}
