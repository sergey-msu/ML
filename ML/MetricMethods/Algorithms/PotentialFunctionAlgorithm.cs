using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;

namespace ML.MetricMethods.Algorithms
{
  public sealed class PotentialFunctionAlgorithm : MetricAlgorithmBase<double[]>, IKernelAlgorithm<double[]>
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
    private double m_H;

    public PotentialFunctionAlgorithm(IMetric<double[]> metric,
                                      IKernel kernel,
                                      double h = 1,
                                      KernelEquipment[] eqps = null)
      : base(metric)
    {
      if (kernel == null)
        throw new MLException("PotentialAlgorithm.ctor(kernel=null)");

      m_Kernel = kernel;
      Eqps = eqps;
      H = h;
    }

    public override string Name { get { return "PF"; } }

    public IKernel Kernel { get { return m_Kernel; } }

    public KernelEquipment[] Eqps
    {
      get { return m_Eqps; }
      set { m_Eqps=value; }
    }

    public double H
    {
      get { return m_H; }
      set
      {
        if (value <= 0)
          throw new MLException("PotentialFunctionAlgorithm.H(value<=0)");

        m_H = value;
      }
    }

    public override double CalculateClassScore(double[] x, Class cls)
    {
      var closeness = 0.0D;
      int idx = -1;

      foreach (var pData in TrainingSample)
      {
        idx++;
        if (!pData.Value.Equals(cls)) continue;

        var h = (m_Eqps==null) ? m_H : m_Eqps[idx].H;
        var g = (m_Eqps==null) ? 1 : m_Eqps[idx].Gamma;
        var r = Metric.Dist(x, pData.Key) / h;

        closeness += (g * m_Kernel.Value(r));
      }

      return closeness;
    }
  }
}
