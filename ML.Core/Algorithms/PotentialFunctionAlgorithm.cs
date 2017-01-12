using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core.Contracts;

namespace ML.Core.Algorithms
{
  public sealed class PotentialFunctionAlgorithm : MetricAlgorithmBase<PotentialFunctionAlgorithm.Params>
  {
    #region Inner

    public class Params
    {
      public Params(float[] gammas, float[] hs)
      {
        if (gammas == null || gammas.Length <= 0)
          throw new ArgumentException("PotentialAlgorithm.Params.ctor(gammas=null|empty)");
        if (hs == null || hs.Length <= 0)
          throw new ArgumentException("PotentialAlgorithm.Params.ctor(hs=null|empty)");
        if (gammas.Length != hs.Length)
          throw new ArgumentException("PotentialAlgorithm.Params.ctor(gammas.length<>hs.length)");

        Gammas = gammas;
        Hs = hs;
      }

      public readonly float[] Gammas;
      public readonly float[] Hs;
    }

    #endregion

    private readonly IKernel m_Kernel;

    public PotentialFunctionAlgorithm(ClassifiedSample classifiedSample,
                              IMetric metric,
                              IKernel kernel,
                              Params pars)
      : base(classifiedSample, metric, pars)
    {
      if (kernel == null)
        throw new ArgumentException("PotentialAlgorithm.ctor(kernel=null)");
      if (Parameters.Gammas.Length != classifiedSample.Count)
        throw new ArgumentException("PotentialAlgorithm.ctor(gammas.length<>sample.length)");

      m_Kernel = kernel;
    }

    public override string ID { get { return "PF"; } }

    public override string Name { get { return "Potential Functions"; } }

    public IKernel Kernel { get { return m_Kernel; } }

    public override float EstimateClose(Point x, Class cls)
    {
      var closeness = 0.0F;
      int idx = -1;

      foreach (var sData in TrainingSample)
      {
        idx++;
        if (sData.Value != cls) continue;

        var r = Metric.Dist(x, sData.Key) / Parameters.Hs[idx];
        closeness += Parameters.Gammas[idx] * m_Kernel.Calculate(r);
      }

      return closeness;
    }

    private float calculateWeight(int i, Point x, Dictionary<Point, float> orderedSample)
    {
      var r = orderedSample.ElementAt(i).Value / Parameters.Hs[i];
      return Parameters.Gammas[i] * m_Kernel.Calculate(r);
    }
  }
}
