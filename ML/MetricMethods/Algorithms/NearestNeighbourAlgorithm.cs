using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;

namespace ML.MetricMethods.Algorithms
{
  /// <summary>
  /// Nearest Neighbour Algorithm
  /// </summary>
  public sealed class NearestNeighbourAlgorithm : OrderedMetricAlgorithmBase<double[]>
  {
    public NearestNeighbourAlgorithm(IMetric<double[]> metric)
      : base(metric)
    {
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public override string Name { get { return "NN"; } }


    /// <summary>
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var minDist = double.MaxValue;
      var result = Class.Unknown;

      foreach (var pData in TrainingSample)
      {
        var dist = Metric.Dist(pData.Key, obj);
        if (dist<minDist)
        {
          minDist = dist;
          result = pData.Value;
        }
      }

      return result;
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
      return i==0 ? 1 : 0;
    }
  }
}
