using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;

namespace ML.MetricalMethods.Algorithms
{
  /// <summary>
  /// Nearest Neighbour Algorithm
  /// </summary>
  public sealed class NearestNeighbourAlgorithm : OrderedMetricAlgorithmBase
  {
    public NearestNeighbourAlgorithm(ClassifiedSample<double[]> classifiedSample, IMetric metric)
      : base(classifiedSample, metric)
    {
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public override string ID { get { return "NN"; } }

    /// <summary>
    /// Algorithm name
    /// </summary>
    public override string Name { get { return "Nearest Neighbour"; } }

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
