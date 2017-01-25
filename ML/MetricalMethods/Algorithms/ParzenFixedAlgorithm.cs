using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;

namespace ML.MetricalMethods.Algorithms
{
  /// <summary>
  /// Parzen Fixed Window Algorithm
  /// </summary>
  public sealed class ParzenFixedAlgorithm : KernelAlgorithmBase
  {
    private float m_H;

    public ParzenFixedAlgorithm(ClassifiedSample classifiedSample,
                                IMetric metric,
                                IKernel kernel,
                                float h)
      : base(classifiedSample, metric, kernel)
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
    public float H
    {
      get { return m_H; }
      set
      {
        if (value <= float.Epsilon)
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
    protected override float CalculateWeight(int i, Point x, Dictionary<Point, float> orderedSample)
    {
      var r = orderedSample.ElementAt(i).Value / m_H;
      return Kernel.Calculate(r);
    }

    /// <summary>
    /// Leave-one-out optimization
    /// </summary>
    public void Train_LOO(float hMin, float hMax, float step)
    {
      var hOpt = float.MaxValue;
      var minErrCnt = int.MaxValue;

      for (float h = hMin; h <= hMax; h += step)
      {
        var errCnt = 0;
        m_H = h;

        for (int i = 0; i < TrainingSample.Count; i++)
        {
          var pData = TrainingSample.ElementAt(i);
          using (var mask = this.ApplySampleMask((p, c, idx) => idx != i))
          {
            var cls = this.Classify(pData.Key);
            if (cls != pData.Value) errCnt++;
          }
        }

        if (errCnt < minErrCnt)
        {
          minErrCnt = errCnt;
          hOpt = h;
        }
      }

      m_H = hOpt;
    }
  }
}
