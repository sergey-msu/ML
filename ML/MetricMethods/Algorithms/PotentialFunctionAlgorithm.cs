using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;

namespace ML.MetricMethods.Algorithms
{
  public sealed class PotentialFunctionAlgorithm : MetricAlgorithmBase<double[]>
  {
    #region Inner

    public struct KernelEquipment
    {
      public static readonly KernelEquipment Trivial = new KernelEquipment(1.0D, 1.0D);

      public KernelEquipment(double gamma, double h)
      {
        Gamma = gamma;
        H = h;
      }

      public double Gamma;
      public double H;
    }

    #endregion

    private readonly IKernel m_Kernel;
    private KernelEquipment[]  m_Eqps;

    public PotentialFunctionAlgorithm(IMetric metric,
                                      IKernel kernel,
                                      KernelEquipment[] eqps)
      : base(metric)
    {
      if (kernel == null)
        throw new MLException("PotentialAlgorithm.ctor(kernel=null)");

      m_Kernel = kernel;
      Eqps = eqps;
    }

    public override string ID { get { return "PF"; } }

    public override string Name { get { return "Potential Functions"; } }

    public IKernel Kernel { get { return m_Kernel; } }

    public KernelEquipment[] Eqps
    {
      get { return m_Eqps; }
      set { m_Eqps=value; }
    }

    public override double CalculateClassScore(double[] x, Class cls)
    {
      var closeness = 0.0D;
      int idx = -1;

      foreach (var pData in TrainingSample)
      {
        idx++;
        if (pData.Value != cls) continue;

        var h = (m_Eqps==null) ? 1 : m_Eqps[idx].H;
        var g = (m_Eqps==null) ? 1 : m_Eqps[idx].Gamma;
        var r = Metric.Dist(x, pData.Key) / h;

        closeness += (g * m_Kernel.Value(r));
      }

      return closeness;
    }
  }
}
