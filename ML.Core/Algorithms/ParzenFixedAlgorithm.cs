using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core.Contracts;


namespace ML.Core.Algorithms
{
  public sealed class ParzenFixedAlgorithm : KernelAlgorithmBase<ParzenFixedAlgorithm.Params>
  {
    #region Inner

    public class Params
    {
      public Params(float h)
      {
        if (h <= float.Epsilon)
          throw new ArgumentException("ParzenFixedAlgorithm.Params.ctor(h<=0)");

        H = h;
      }

      public readonly float H;
    }

    #endregion

    public ParzenFixedAlgorithm(ClassifiedSample classifiedSample,
                                IMetric metric,
                                IKernel kernel,
                                Params pars)
      : base(classifiedSample, metric, kernel, pars)
    {
    }

    public override string ID { get { return "PFW"; } }

    public override string Name { get { return "Parzen Fixed Width Window"; } }

    protected override float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample)
    {
      var r = orderedSample.ElementAt(i).Value / Parameters.H;
      return Kernel.Calculate(r);
    }
  }
}
