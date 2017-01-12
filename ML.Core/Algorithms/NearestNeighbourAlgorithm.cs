using System;
using System.Collections.Generic;
using ML.Core.Contracts;


namespace ML.Core.Algorithms
{
  public sealed class NearestNeighbourAlgorithm : OrderedMetricAlgorithmBase<NearestNeighbourAlgorithm.Params>
  {
    #region Inner

    public class Params {}

    #endregion

    public NearestNeighbourAlgorithm(ClassifiedSample classifiedSample, IMetric metric, Params pars=null)
      : base(classifiedSample, metric, pars)
    {
    }

    public override string ID { get { return "NN"; } }

    public override string Name { get { return "Nearest K Neighbours"; } }

    protected override float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample)
    {
      return i==0 ? 1 : 0;
    }
  }
}
