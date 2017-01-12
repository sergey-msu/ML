using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core.Contracts;

namespace ML.Core.Algorithms
{
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

    public override string ID { get { return "NNK"; } }

    public override string Name { get { return "Nearest K Neighbour(s)"; } }

    protected override float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample)
    {
      return (i < Parameters.K) ? 1 : 0;
    }
  }
}
