using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core.Contracts;

namespace ML.Core.Algorithms
{
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

    public override string ID { get { return "WNNK"; } }

    public override string Name { get { return "Weighted Nearest K Neighbours"; } }

    protected override float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample)
    {
      return i < Parameters.K ? Parameters.Weights[i] : 0;
    }
  }
}
