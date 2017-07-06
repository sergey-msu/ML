using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;

namespace ML.MetricMethods.Algorithms
{
  /// <summary>
  /// Parzen Fixed Window Algorithm
  /// </summary>
  public sealed class ParzenFixedAlgorithm : KernelAlgorithmBase
  {
    private double m_H;

    public ParzenFixedAlgorithm(IMetric metric,
                                IKernel kernel,
                                double h)
      : base(metric, kernel)
    {
      H = h;
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
    /// Window width
    /// </summary>
    public double H
    {
      get { return m_H; }
      set
      {
        if (value <= double.Epsilon)
          throw new MLException("ParzenFixedAlgorithm.H(value<=0)");

        m_H = value;
      }
    }

    /// <summary>
    /// Calculate 'weight' - a contribution of training point (i-th from ordered training sample)
    /// to closeness of test point to its class
    /// </summary>
    /// <param name="i">Point number in ordered training sample</param>
    /// <param name="x">Test point</param>
    /// <param name="orderedSample">Ordered training sample</param>
    protected override double CalculateWeight(int i, double[] x, Dictionary<double[], double> orderedSample)
    {
      var r = orderedSample.ElementAt(i).Value / m_H;
      return Kernel.Value(r);
    }

    /// <summary>
    /// Leave-one-out optimization
    /// </summary>
    public void OptimizeLOO(double hMin, double hMax, double step)
    {
      var hOpt = double.MaxValue;
      var minErrCnt = int.MaxValue;

      for (double h = hMin; h <= hMax; h += step)
      {
        var errCnt = 0;
        H = h;

        var initSample = TrainingSample;

        for (int i=0; i<initSample.Count; i++)
        {
          var pData = initSample.ElementAt(i);
          var looSample  = initSample.ApplyMask((p, c, idx) => idx != i);
          TrainingSample = looSample;

          var predClass = this.Predict(pData.Key);
          var realClass = pData.Value;
          if (predClass != realClass) errCnt++;

          TrainingSample = initSample;
        }

        if (errCnt < minErrCnt)
        {
          minErrCnt = errCnt;
          hOpt = h;
        }
      }

      H = hOpt;
    }
  }
}
