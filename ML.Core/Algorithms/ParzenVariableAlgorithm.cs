using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core.Contracts;

namespace ML.Core.Algorithms
{
  public sealed class ParzenVariableAlgorithm : KernelAlgorithmBase<ParzenVariableAlgorithm.Params>
  {
    #region Inner

    public class Params
    {
      public Params(int k)
      {
        if (k <= 0)
          throw new ArgumentException("ParzenVariableAlgorithm.Params.ctor(k<=0)");

        K = k;
      }

      public readonly int K;
    }

    #endregion

    public ParzenVariableAlgorithm(ClassifiedSample classifiedSample,
                                   IMetric metric,
                                   IKernel kernel,
                                   Params pars)
      : base(classifiedSample, metric, kernel, pars)
    {
      if (pars.K < 0 || pars.K+1 >= classifiedSample.Count)
        throw new ArgumentException("ParzenFixedAlgorithm.ctor(k<0|k+1>=classifiedSample.Count)");
    }

    public override string ID { get { return "PVW"; } }

    public override string Name { get { return "Parzen Variable Width Window"; } }

    protected override float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample)
    {
      var r = orderedSample.ElementAt(i).Value / orderedSample.ElementAt(Parameters.K+1).Value;
      return Kernel.Calculate(r);
    }
  }
}
