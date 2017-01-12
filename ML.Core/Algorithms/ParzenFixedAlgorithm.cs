using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core.Contracts;


namespace ML.Core.Algorithms
{
  /// <summary>
  /// Parzen Fixed Window Algorithm
  /// </summary>
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

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public override string ID { get { return "PFW"; } }

    /// <summary>
    /// Algorithm name
    /// </summary>
    public override string Name { get { return "Parzen Fixed Width Window"; } }

    /// <summary>
    /// Calculate 'weight' - a contribution of training point (i-th from ordered training sample)
    /// to closeness of test point to its class
    /// </summary>
    /// <param name="i">Point number in ordered training sample</param>
    /// <param name="x">Test point</param>
    /// <param name="orderedSample">Ordered training sample</param>
    protected override float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample)
    {
      var r = orderedSample.ElementAt(i).Value / Parameters.H;
      return Kernel.Calculate(r);
    }
  }
}
